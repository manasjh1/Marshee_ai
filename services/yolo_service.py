import os
import cv2
import numpy as np
import base64
import logging
import time
from typing import Dict, Any, Optional, List
from PIL import Image
from io import BytesIO
import torch
from ultralytics import YOLO
from modals.chat import YOLORequest, YOLODetectionResult

logger = logging.getLogger(__name__)

class YOLOService:
    def __init__(self):
        # Paths to your trained .pt models
        self.breed_model_path = os.getenv("BREED_MODEL_PATH", "models/dog_breed_yolov11.pt")
        self.disease_model_path = os.getenv("DISEASE_MODEL_PATH", "models/dog_disease_yolov11.pt")
        
        # Model instances
        self.breed_model = None
        self.disease_model = None
        
        # Confidence thresholds
        self.breed_confidence_threshold = float(os.getenv("BREED_CONFIDENCE_THRESHOLD", "0.5"))
        self.disease_confidence_threshold = float(os.getenv("DISEASE_CONFIDENCE_THRESHOLD", "0.6"))
        
        # Load your trained models
        self._load_models()

        logger.info("YOLOv11 Service initialized")

    def _load_models(self):
        """Load your trained YOLOv11 models"""
        try:
            # Load breed detection model
            if os.path.exists(self.breed_model_path):
                self.breed_model = YOLO(self.breed_model_path)
                logger.info(f"Loaded breed detection model: {self.breed_model_path}")
                logger.info(f"Breed model classes: {list(self.breed_model.names.values())}")
            else:
                logger.error(f"Breed model not found: {self.breed_model_path}")
            
            # Load disease detection model
            if os.path.exists(self.disease_model_path):
                self.disease_model = YOLO(self.disease_model_path)
                logger.info(f"Loaded disease detection model: {self.disease_model_path}")
                logger.info(f"Disease model classes: {list(self.disease_model.names.values())}")
            else:
                logger.error(f"Disease model not found: {self.disease_model_path}")

        except Exception as e:
            logger.error(f"Error loading YOLOv11 models: {e}")

    def detect_breed(self, request: YOLORequest) -> YOLODetectionResult:
        """
        Detect dog breed using your trained YOLOv11 model
        Returns text results as you requested
        """
        start_time = time.time()
        
        try:
            if not self.breed_model:
                raise Exception("Breed detection model not loaded")
            
            # Decode base64 image to format YOLOv11 can process
            image_path = self._decode_base64_image(request.image_data)
            
            # Run YOLOv11 inference
            results = self.breed_model(image_path, conf=self.breed_confidence_threshold, verbose=False)
            
            # Clean up temp file
            if os.path.exists(image_path):
                os.remove(image_path)
            
            # Extract results and convert to text
            if len(results) > 0 and len(results[0].boxes) > 0:
                # Get the best detection
                result = results[0]
                best_box = result.boxes[0]  # Highest confidence detection
                
                # Extract information
                confidence = float(best_box.conf[0])
                class_id = int(best_box.cls[0])
                breed_name = self.breed_model.names[class_id]  # Get class name from model
                
                # Create text result
                text_result = self._format_breed_result(breed_name, confidence)
                
                processing_time = time.time() - start_time
                
                return YOLODetectionResult(
                    model_type="breed",
                    confidence=confidence,
                    detected_class=breed_name,
                    text_result=text_result,
                    additional_info={
                        "class_id": class_id,
                        "breed_characteristics": self._get_breed_basic_info(breed_name)
                    },
                    processing_time=processing_time
                )
            else:
                # No detection found
                text_result = "No clear dog breed could be detected in the image. Please provide a clearer photo showing the full dog."
                
                processing_time = time.time() - start_time
                return YOLODetectionResult(
                    model_type="breed",
                    confidence=0.0,
                    detected_class="unknown",
                    text_result=text_result,
                    additional_info={},
                    processing_time=processing_time
                )
                
        except Exception as e:
            logger.error(f"Breed detection error: {e}")
            processing_time = time.time() - start_time
            
            return YOLODetectionResult(
                model_type="breed",
                confidence=0.0,
                detected_class="error",
                text_result=f"Error processing image: {str(e)}",
                additional_info={},
                processing_time=processing_time
            )

    def detect_disease(self, request: YOLORequest) -> YOLODetectionResult:
        """
        Detect skin conditions/diseases using your trained YOLOv11 model
        Returns text results as you requested
        """
        start_time = time.time()
        
        try:
            if not self.disease_model:
                raise Exception("Disease detection model not loaded")
            
            # Decode base64 image
            image_path = self._decode_base64_image(request.image_data)
            
            # Run YOLOv11 inference
            results = self.disease_model(image_path, conf=self.disease_confidence_threshold, verbose=False)
            
            # Clean up temp file
            if os.path.exists(image_path):
                os.remove(image_path)
            
            # Extract results and convert to text
            if len(results) > 0 and len(results[0].boxes) > 0:
                # Get the best detection
                result = results[0]
                best_box = result.boxes[0]  # Highest confidence detection
                
                # Extract information
                confidence = float(best_box.conf[0])
                class_id = int(best_box.cls[0])
                condition_name = self.disease_model.names[class_id]  # Get class name from model
                
                # Create text result with health guidance
                text_result = self._format_disease_result(condition_name, confidence)
                
                processing_time = time.time() - start_time
                
                return YOLODetectionResult(
                    model_type="disease",
                    confidence=confidence,
                    detected_class=condition_name,
                    text_result=text_result,
                    additional_info={
                        "class_id": class_id,
                        "severity_assessment": self._assess_condition_severity(condition_name, confidence),
                        "immediate_care_tips": self._get_immediate_care_tips(condition_name)
                    },
                    processing_time=processing_time
                )
            else:
                # No disease detected - healthy skin
                text_result = "Good news! No skin conditions detected. Your dog's skin appears healthy in this image."
                
                processing_time = time.time() - start_time
                return YOLODetectionResult(
                    model_type="disease",
                    confidence=1.0,  # High confidence for "normal"
                    detected_class="healthy",
                    text_result=text_result,
                    additional_info={
                        "health_status": "normal",
                        "care_tips": "Continue regular grooming and monitoring"
                    },
                    processing_time=processing_time
                )
                
        except Exception as e:
            logger.error(f"Disease detection error: {e}")
            processing_time = time.time() - start_time
            
            return YOLODetectionResult(
                model_type="disease",
                confidence=0.0,
                detected_class="error",
                text_result=f"Error processing image: {str(e)}",
                additional_info={},
                processing_time=processing_time
            )

    def _decode_base64_image(self, base64_data: str) -> str:
        """
        Decode base64 image and save temporarily for YOLOv11 processing
        YOLOv11 works best with file paths or PIL Images
        """
        try:
            # Remove data URL prefix if present (data:image/jpeg;base64,)
            if "," in base64_data:
                base64_data = base64_data.split(",")[1]
            
            # Decode base64
            image_bytes = base64.b64decode(base64_data)
            
            # Convert to PIL Image
            pil_image = Image.open(BytesIO(image_bytes))
            
            # Convert to RGB if needed
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Save temporarily for YOLOv11
            temp_path = f"temp_image_{int(time.time())}.jpg"
            pil_image.save(temp_path)
            
            return temp_path  # Return path for YOLOv11
            
        except Exception as e:
            logger.error(f"Error decoding image: {e}")
            raise

    def _format_breed_result(self, breed_name: str, confidence: float) -> str:
        """Format breed detection result as text"""
        
        confidence_percent = confidence * 100
        
        if confidence > 0.8:
            confidence_text = "very confident"
        elif confidence > 0.6:
            confidence_text = "confident"
        elif confidence > 0.4:
            confidence_text = "somewhat confident"
        else:
            confidence_text = "not very confident"
        
        result_text = f"I am {confidence_text} ({confidence_percent:.1f}%) that this is a {breed_name}. "
        
        # Add breed-specific information if available
        breed_info = self._get_breed_basic_info(breed_name)
        if breed_info:
            result_text += f"\n\n{breed_info}"
        
        return result_text

    def _format_disease_result(self, condition_name: str, confidence: float) -> str:
        """Format disease detection result as text with guidance"""
        
        confidence_percent = confidence * 100
        
        result_text = f"I detected signs of {condition_name} with {confidence_percent:.1f}% confidence.\n\n"
        
        # Add condition-specific guidance
        guidance = self._get_condition_guidance(condition_name)
        result_text += guidance
        
        return result_text

    def _get_breed_basic_info(self, breed_name: str) -> str:
        """Get basic information about the detected breed"""
        # You can customize this based on the breeds your model can detect
        breed_info = {
            "Golden Retriever": "Golden Retrievers are friendly, intelligent, and devoted dogs. They need regular exercise and grooming.",
            "Labrador": "Labradors are outgoing, active, and friendly. They require plenty of exercise and mental stimulation.",
            "German Shepherd": "German Shepherds are confident, courageous, and smart. They need lots of exercise and training.",
            "Bulldog": "Bulldogs are calm, courageous, and friendly. They have moderate exercise needs but watch for breathing issues.",
            "Beagle": "Beagles are curious, friendly, and merry. They need regular exercise and mental stimulation.",
            "Rottweiler": "Rottweilers are confident, fearless, and good-natured. They need consistent training and socialization.",
            "Poodle": "Poodles are intelligent, active, and elegant. They require regular grooming and mental stimulation.",
            "Siberian Husky": "Huskies are outgoing, mischievous, and loyal. They have very high exercise needs and love cold weather.",
            # Add more breeds based on your model's classes
        }
        
        return breed_info.get(breed_name, f"{breed_name} is a wonderful breed! Regular exercise, proper nutrition, and veterinary care are important.")

    def _get_condition_guidance(self, condition_name: str) -> str:
        """Get guidance for detected skin conditions"""
        # Customize based on the conditions your model can detect
        condition_guidance = {
            "Hot Spot": "Hot spots are painful, red, and inflamed areas. Keep the area clean and dry, prevent licking, and consult your vet within 24-48 hours for proper treatment.",
            
            "Allergic Dermatitis": "This appears to be an allergic skin reaction. Try to identify and remove the allergen, give your dog a gentle bath, and schedule a vet visit within a few days.",
            
            "Fungal Infection": "Fungal infections can spread to other pets. Keep the area dry, isolate your dog if possible, and see a vet within 1-2 days for antifungal treatment.",
            
            "Bacterial Infection": "Bacterial skin infections require antibiotic treatment. Keep the area clean and contact your veterinarian soon for proper medication.",
            
            "Mange": "Mange is caused by mites and is very uncomfortable. This requires immediate veterinary attention for proper diagnosis and treatment.",
            
            "Eczema": "Eczema can be caused by various factors including allergies or dry skin. Consult your vet for proper diagnosis and treatment plan.",
            
            "Healthy": "Your dog's skin looks healthy! Continue regular grooming and watch for any changes in skin condition."
        }
        
        return condition_guidance.get(condition_name, f"I detected {condition_name}. Please consult with your veterinarian for proper diagnosis and treatment recommendations.")

    def _assess_condition_severity(self, condition_name: str, confidence: float) -> str:
        """Assess the severity and urgency of the detected condition"""
        
        # Severity mapping based on condition
        severity_map = {
            "Hot Spot": "Moderate - Needs attention within 24-48 hours",
            "Allergic Dermatitis": "Mild to Moderate - Schedule vet visit within a few days", 
            "Fungal Infection": "Moderate - See vet within 1-2 days",
            "Bacterial Infection": "Moderate - Veterinary treatment needed soon",
            "Mange": "High - Requires immediate veterinary attention",
            "Eczema": "Mild to Moderate - Monitor and consult vet if worsens",
            "Healthy": "None - Continue regular care"
        }
        
        base_severity = severity_map.get(condition_name, "Unknown - Consult veterinarian")
        
        # Adjust based on confidence
        if confidence < 0.6:
            return f"{base_severity} (Low detection confidence - consider getting a second opinion or clearer image)"
        
        return base_severity

    def _get_immediate_care_tips(self, condition_name: str) -> List[str]:
        """Get immediate care tips for detected conditions"""
        care_tips = {
            "Hot Spot": [
                "Keep the area clean and dry",
                "Prevent your dog from licking or scratching",
                "Apply a cool, damp compress for relief",
                "Trim hair around the affected area if possible"
            ],
            "Allergic Dermatitis": [
                "Identify and remove potential allergens",
                "Give your dog a gentle, cool bath",
                "Avoid harsh chemicals or perfumed products",
                "Monitor for worsening symptoms"
            ],
            "Fungal Infection": [
                "Keep the affected area dry",
                "Avoid spreading to other pets",
                "Clean and disinfect your dog's bedding",
                "Wash your hands after touching the area"
            ],
            "Bacterial Infection": [
                "Keep the area clean with gentle cleaning",
                "Prevent scratching or licking",
                "Monitor for spreading or worsening",
                "Note any discharge or odor for vet consultation"
            ],
            "Mange": [
                "Isolate your dog from other pets temporarily",
                "Keep bedding and toys separate",
                "Schedule immediate vet appointment",
                "Avoid home remedies without vet guidance"
            ]
        }
        
        return care_tips.get(condition_name, [
            "Monitor the condition closely",
            "Keep the area clean",
            "Consult with your veterinarian",
            "Document any changes with photos"
        ])

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        info = {
            "breed_model": {
                "loaded": self.breed_model is not None,
                "path": self.breed_model_path,
                "classes": list(self.breed_model.names.values()) if self.breed_model else [],
                "class_count": len(self.breed_model.names) if self.breed_model else 0
            },
            "disease_model": {
                "loaded": self.disease_model is not None,
                "path": self.disease_model_path,
                "classes": list(self.disease_model.names.values()) if self.disease_model else [],
                "class_count": len(self.disease_model.names) if self.disease_model else 0
            }
        }
        return info

    def cleanup_temp_files(self):
        """Clean up temporary image files"""
        try:
            import glob
            temp_files = glob.glob("temp_image_*.jpg")
            for file in temp_files:
                if os.path.exists(file):
                    os.remove(file)
                    logger.debug(f"Cleaned up temp file: {file}")
        except Exception as e:
            logger.warning(f"Error cleaning temp files: {e}")