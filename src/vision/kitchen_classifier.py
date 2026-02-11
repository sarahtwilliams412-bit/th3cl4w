"""Kitchen scene vision backbone for object detection and state classification.

Pre-trained on NVIDIA PhysicalAI-Robotics-Kitchen-Sim-Demos camera views,
this module provides fast, local object detection and state classification
without requiring Gemini API calls.

Two models:
1. Object state classifier: Is the cabinet open/closed? Gripper empty/holding?
2. Object detector: Where are kitchen objects in the frame?

Usage:
    classifier = KitchenStateClassifier()
    states = classifier.classify(frame_bgr)
    # states = {"cabinet": "open", "gripper": "holding", ...}

    detector = KitchenObjectDetector()
    detections = detector.detect(frame_bgr)
    # detections = [{"label": "mug", "bbox": [x, y, w, h], "confidence": 0.9}, ...]
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Kitchen object state labels (from NVIDIA dataset task categories)
STATE_LABELS = [
    "cabinet_open", "cabinet_closed",
    "drawer_open", "drawer_closed",
    "microwave_open", "microwave_closed",
    "fridge_open", "fridge_closed",
    "oven_open", "oven_closed",
    "gripper_empty", "gripper_holding",
    "stove_on", "stove_off",
    "sink_on", "sink_off",
]

# Object classes relevant to kitchen manipulation
KITCHEN_OBJECT_CLASSES = [
    "mug", "bowl", "plate", "pot", "pan",
    "spatula", "whisk", "tongs", "knife",
    "cutting_board", "food_container",
    "bottle", "can", "kettle", "colander",
    "cabinet_handle", "drawer_handle",
    "stove_knob", "faucet_handle",
]

# Model paths
_DEFAULT_CLASSIFIER_PATH = Path("data/models/kitchen_state_classifier")
_DEFAULT_DETECTOR_PATH = Path("data/models/kitchen_object_detector")


class KitchenStateClassifier:
    """Classifies kitchen object states from camera images.

    Uses a lightweight vision model (MobileNetV3 or EfficientNet-B0)
    fine-tuned on NVIDIA Kitchen-Sim-Demos frames to classify states
    like cabinet open/closed, gripper holding/empty, etc.

    This provides real-time (>30fps) state estimation without API calls,
    feeding into the world model service.
    """

    def __init__(self, model_path: Optional[Path] = None, device: str = "auto"):
        self._model_path = model_path or _DEFAULT_CLASSIFIER_PATH
        self._device = device
        self._model = None
        self._transform = None
        self._loaded = False

    def _load_model(self) -> bool:
        """Lazy-load the state classification model."""
        if self._loaded:
            return True

        model_dir = Path(self._model_path)

        # Try loading fine-tuned model
        if model_dir.exists():
            try:
                import torch
                weights_path = model_dir / "model.pt"
                if weights_path.exists():
                    self._model = torch.load(weights_path, map_location="cpu")
                    self._model.eval()
                    self._loaded = True
                    logger.info("Kitchen state classifier loaded from %s", model_dir)
                    return True
            except Exception as e:
                logger.warning("Failed to load fine-tuned classifier: %s", e)

        # Fall back to pre-trained MobileNetV3 (will need fine-tuning)
        try:
            import torch
            import torchvision.models as models
            import torchvision.transforms as transforms

            if self._device == "auto":
                self._device = "cuda" if torch.cuda.is_available() else "cpu"

            self._model = models.mobilenet_v3_small(weights="DEFAULT")
            # Replace classifier head for our state labels
            num_classes = len(STATE_LABELS)
            self._model.classifier[-1] = torch.nn.Linear(
                self._model.classifier[-1].in_features, num_classes
            )
            self._model.to(self._device)
            self._model.eval()

            self._transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

            self._loaded = True
            logger.info(
                "Kitchen state classifier initialized (MobileNetV3, %d classes, "
                "requires fine-tuning on kitchen data)",
                num_classes,
            )
            return True

        except ImportError:
            logger.warning("torchvision not available for kitchen classifier")
            return False

    def classify(self, frame_bgr: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Classify kitchen object states in a camera frame.

        Args:
            frame_bgr: HxWx3 uint8 BGR image (OpenCV format)

        Returns:
            Dict mapping state names to confidence scores.
            Example: {"cabinet_open": 0.85, "gripper_holding": 0.92, ...}
        """
        if not self._load_model():
            return {}

        import torch
        import cv2

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        input_tensor = self._transform(frame_rgb).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._model(input_tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        results = {}
        for i, label in enumerate(STATE_LABELS):
            if i < len(probs):
                results[label] = float(probs[i])

        return results

    def get_binary_states(
        self, frame_bgr: np.ndarray, threshold: float = 0.5,
    ) -> Dict[str, str]:
        """Get binary state decisions (open/closed, on/off, etc.).

        Returns:
            Dict mapping fixture names to their state.
            Example: {"cabinet": "open", "gripper": "holding", "stove": "off"}
        """
        raw = self.classify(frame_bgr)
        states = {}

        # Pair up state labels
        pairs = [
            ("cabinet", "cabinet_open", "cabinet_closed"),
            ("drawer", "drawer_open", "drawer_closed"),
            ("microwave", "microwave_open", "microwave_closed"),
            ("fridge", "fridge_open", "fridge_closed"),
            ("oven", "oven_open", "oven_closed"),
            ("gripper", "gripper_holding", "gripper_empty"),
            ("stove", "stove_on", "stove_off"),
            ("sink", "sink_on", "sink_off"),
        ]

        for name, pos_label, neg_label in pairs:
            pos_conf = raw.get(pos_label, 0.0)
            neg_conf = raw.get(neg_label, 0.0)
            if pos_conf > neg_conf:
                states[name] = pos_label.split("_", 1)[1]  # "open", "holding", "on"
            else:
                states[name] = neg_label.split("_", 1)[1]  # "closed", "empty", "off"

        return states


class KitchenObjectDetector:
    """Detects kitchen objects in camera frames.

    Uses a lightweight detection model fine-tuned on kitchen scenes
    from the NVIDIA dataset. Provides fast pre-filtering before
    optional Gemini confirmation for low-confidence detections.
    """

    def __init__(self, model_path: Optional[Path] = None, device: str = "auto"):
        self._model_path = model_path or _DEFAULT_DETECTOR_PATH
        self._device = device
        self._model = None
        self._loaded = False

    def _load_model(self) -> bool:
        """Lazy-load the object detection model."""
        if self._loaded:
            return True

        model_dir = Path(self._model_path)
        if model_dir.exists():
            try:
                import torch
                weights_path = model_dir / "model.pt"
                if weights_path.exists():
                    self._model = torch.load(weights_path, map_location="cpu")
                    self._loaded = True
                    logger.info("Kitchen detector loaded from %s", model_dir)
                    return True
            except Exception as e:
                logger.warning("Failed to load detector: %s", e)

        logger.info(
            "Kitchen object detector not available (no model at %s). "
            "Train with: python -m scripts.train_kitchen_detector",
            model_dir,
        )
        return False

    def detect(
        self,
        frame_bgr: np.ndarray,
        confidence_threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Detect kitchen objects in a camera frame.

        Args:
            frame_bgr: HxWx3 uint8 BGR image
            confidence_threshold: Minimum confidence to report

        Returns:
            List of detections, each: {"label", "bbox", "confidence"}
            bbox format: [x, y, width, height] in pixels
        """
        if not self._load_model():
            return []

        import torch
        import cv2

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Model-specific inference
        if hasattr(self._model, "predict"):
            results = self._model.predict(frame_rgb, conf=confidence_threshold)
            detections = []
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = (
                        KITCHEN_OBJECT_CLASSES[cls_id]
                        if cls_id < len(KITCHEN_OBJECT_CLASSES)
                        else f"class_{cls_id}"
                    )
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append({
                        "label": label,
                        "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                        "confidence": float(box.conf[0]),
                    })
            return detections

        return []

    def detect_and_filter(
        self,
        frame_bgr: np.ndarray,
        target_objects: Optional[List[str]] = None,
        confidence_threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Detect and filter to only specified object types.

        Args:
            frame_bgr: Camera frame
            target_objects: Only return these object types (None = all)
            confidence_threshold: Minimum confidence

        Returns:
            Filtered list of detections
        """
        detections = self.detect(frame_bgr, confidence_threshold)
        if target_objects is not None:
            target_set = set(target_objects)
            detections = [d for d in detections if d["label"] in target_set]
        return detections
