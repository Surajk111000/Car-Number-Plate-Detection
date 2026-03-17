import base64
import logging
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

if TYPE_CHECKING:
    from src.ocr import PlateOCR
    from src.plate_detector import PlateDetector

logger = logging.getLogger(__name__)


# Pydantic response models for OpenAPI documentation
class BoundingBox(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    width: int
    height: int


class OCRSegment(BaseModel):
    text: str
    confidence: float


class OCRResult(BaseModel):
    raw_text: str
    cleaned_text: str
    avg_confidence: float
    is_valid_plate: bool
    segments: List[OCRSegment]


class PlateDetection(BaseModel):
    plate_index: int
    detection_score: float
    bounding_box: BoundingBox
    ocr: Optional[OCRResult] = None


class ImageInfo(BaseModel):
    width: int
    height: int
    channels: int


class PredictionSummary(BaseModel):
    detections: int
    returned_plates: int
    inference_ms: float


class PredictionResponse(BaseModel):
    success: bool
    image: ImageInfo
    summary: PredictionSummary
    plates: List[PlateDetection]
    visualization_base64: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    ocr_loaded: bool
    ocr_enabled: bool
    startup_error: Optional[str]
    config: Dict[str, Any]
    
    class Config:
        protected_namespaces = ()


class IndexResponse(BaseModel):
    message: str
    endpoints: Dict[str, str]


@dataclass
class ServiceConfig:
    model_path: str
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.5
    padding: int = 10
    enable_ocr: bool = True
    ocr_languages: Tuple[str, ...] = ("en",)
    ocr_use_gpu: Optional[bool] = None
    max_content_length: int = 8 * 1024 * 1024


class PlateRecognitionService:
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.detector: Optional["PlateDetector"] = None
        self.ocr: Optional["PlateOCR"] = None
        self.startup_error: Optional[str] = None
        self._ocr_loaded = False
        self._load_components()

    def _load_components(self) -> None:
        try:
            logger.info("Loading specialized license plate detector...")
            from src.license_plate_detector import LicensePlateTextDetector

            self.detector = LicensePlateTextDetector(
                confidence_threshold=self.config.confidence_threshold
            )

            logger.info("✅ License plate text detector loaded successfully")
            # Note: OCR will be lazy-loaded on first prediction for memory efficiency

            self.startup_error = None
            logger.info("Plate recognition service initialized successfully")
        except Exception as exc:
            self.startup_error = str(exc)
            logger.exception("Failed to initialize service")

    def _lazy_load_ocr(self) -> None:
        """Lazy load OCR on first prediction request for memory efficiency"""
        if self._ocr_loaded or not self.config.enable_ocr:
            return
        
        try:
            logger.info("Lazy-loading OCR component...")
            from src.ocr import PlateOCR
            
            ocr_kwargs: Dict[str, Any] = {
                "languages": list(self.config.ocr_languages),
            }
            if self.config.ocr_use_gpu is not None:
                ocr_kwargs["use_gpu"] = self.config.ocr_use_gpu
            self.ocr = PlateOCR(**ocr_kwargs)
            self._ocr_loaded = True
            logger.info("✅ OCR component loaded successfully")
        except ImportError as ie:
            logger.warning(f"OCR not available: {ie}. Predictions will work without OCR.")
            self.ocr = None
            self._ocr_loaded = True
        except Exception as ocr_exc:
            logger.warning(f"Failed to initialize OCR: {ocr_exc}. Predictions will work without OCR.")
            self.ocr = None
            self._ocr_loaded = True

    def is_ready(self) -> bool:
        return self.detector is not None and self.startup_error is None

    def health(self) -> Dict[str, Any]:
        return {
            "status": "ready" if self.is_ready() else "not_ready",
            "model_loaded": self.detector is not None,
            "ocr_loaded": self.ocr is not None if self.config.enable_ocr else False,
            "ocr_enabled": self.config.enable_ocr,
            "startup_error": self.startup_error,
            "config": {
                "confidence_threshold": self.config.confidence_threshold,
                "nms_threshold": self.config.nms_threshold,
                "padding": self.config.padding,
                "ocr_languages": list(self.config.ocr_languages),
            },
        }

    def predict(self, image: np.ndarray, include_visualization: bool = False) -> Dict[str, Any]:
        if not self.is_ready() or self.detector is None:
            raise RuntimeError(self.startup_error or "Service is not ready")

        # Lazy load OCR if needed
        if self.config.enable_ocr:
            self._lazy_load_ocr()

        start_time = time.perf_counter()
        detections = self.detector.detect(image)
        plate_regions = self.detector.get_plate_regions(image, detections)

        plates: List[Dict[str, Any]] = []
        for idx, ((plate_image, (x_min, y_min, x_max, y_max)), score) in enumerate(
            zip(plate_regions, detections["detection_scores"]),
            start=1,
        ):
            plate_payload: Dict[str, Any] = {
                "plate_index": idx,
                "detection_score": float(score),
                "bounding_box": {
                    "x_min": int(x_min),
                    "y_min": int(y_min),
                    "x_max": int(x_max),
                    "y_max": int(y_max),
                    "width": int(x_max - x_min),
                    "height": int(y_max - y_min),
                },
            }

            # Always try real OCR if available
            if self.ocr is not None:
                try:
                    logger.info(f"Extracting OCR for plate {idx}, image shape: {plate_image.shape if hasattr(plate_image, 'shape') else 'unknown'}")
                    # Use lower confidence threshold for better detection on various plate formats
                    # including Indian, European, and other formats
                    ocr_result = self.ocr.extract_text(plate_image, confidence_threshold=0.1)
                    logger.info(f"OCR result for plate {idx}: {ocr_result}")
                    plate_payload["ocr"] = {
                        "raw_text": ocr_result.get("full_text", ""),
                        "cleaned_text": ocr_result.get("cleaned_text", ""),
                        "avg_confidence": float(ocr_result.get("avg_confidence", 0.0)),
                        "is_valid_plate": bool(ocr_result.get("is_valid_plate", False)),
                        "segments": [
                            {
                                "text": text,
                                "confidence": float(confidence),
                            }
                            for text, confidence in zip(
                                ocr_result.get("text", []),
                                ocr_result.get("confidence", []),
                            )
                        ],
                    }
                except Exception as ocr_error:
                    logger.error(f"OCR extraction failed for plate {idx}: {type(ocr_error).__name__}: {ocr_error}", exc_info=True)
                    plate_payload["ocr"] = {
                        "raw_text": "",
                        "cleaned_text": "",
                        "avg_confidence": 0.0,
                        "is_valid_plate": False,
                        "segments": [],
                    }

            plates.append(plate_payload)

        inference_ms = round((time.perf_counter() - start_time) * 1000, 2)
        response: Dict[str, Any] = {
            "success": True,
            "image": {
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
                "channels": int(image.shape[2]) if len(image.shape) == 3 else 1,
            },
            "summary": {
                "detections": int(detections["num_detections"]),
                "returned_plates": len(plates),
                "inference_ms": inference_ms,
            },
            "plates": plates,
        }

        if include_visualization:
            visualization = self.detector.visualize(image, detections)
            success, encoded = cv2.imencode(".jpg", visualization)
            if success:
                response["visualization_base64"] = base64.b64encode(encoded.tobytes()).decode("utf-8")

        return response


def _parse_bool(value: Optional[str], default: Optional[bool] = None) -> Optional[bool]:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _service_config_from_env() -> ServiceConfig:
    languages = tuple(
        lang.strip() for lang in os.getenv("OCR_LANGUAGES", "en").split(",") if lang.strip()
    )
    
    return ServiceConfig(
        model_path="pretrained",  # Using YOLOv8 pretrained
        confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.5")),
        nms_threshold=float(os.getenv("NMS_THRESHOLD", "0.5")),
        padding=int(os.getenv("PADDING", "10")),
        enable_ocr=_parse_bool(os.getenv("ENABLE_OCR"), True) is not False,
        ocr_languages=languages or ("en",),
        ocr_use_gpu=_parse_bool(os.getenv("OCR_USE_GPU"), None),
        max_content_length=int(os.getenv("MAX_CONTENT_LENGTH", str(8 * 1024 * 1024))),
    )


def _decode_uploaded_image(file_bytes: bytes) -> np.ndarray:
    if not file_bytes:
        raise ValueError("Uploaded file is empty")

    image_array = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode image. Please upload a valid JPG or PNG file")
    return image


def create_app() -> FastAPI:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

    app = FastAPI(
        title="Car Plate Recognition API",
        description="Detect and recognize car license plates using TensorFlow and OCR",
        version="1.0.0",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    config = _service_config_from_env()
    service = PlateRecognitionService(config)

    @app.get("/", response_model=IndexResponse)
    def index():
        return {
            "message": "Car plate recognition API is running",
            "endpoints": {
                "health": "/health",
                "predict": "/predict",
                "docs": "/docs",
                "redoc": "/redoc",
            },
        }

    @app.get("/health", response_model=HealthResponse)
    def health():
        status_code = 200 if service.is_ready() else 503
        health_data = service.health()
        if not service.is_ready():
            raise HTTPException(status_code=status_code, detail=health_data)
        return health_data

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(
        image: UploadFile = File(..., description="JPG or PNG image file"),
        include_visualization: bool = Form(False, description="Include visualization in response"),
    ):
        if not service.is_ready():
            raise HTTPException(
                status_code=503,
                detail=service.startup_error or "Service not ready",
            )

        try:
            file_bytes = await image.read()
            img = _decode_uploaded_image(file_bytes)
            result = service.predict(img, include_visualization=include_visualization)
            return result
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.exception("Prediction failed")
            raise HTTPException(status_code=500, detail=str(exc))

    return app


app = create_app()
