"""
Keras OCR Standalone Module - Simplified Version
Optimized for Python 3.9+ with TensorFlow 2.x
"""

import cv2
import numpy as np
import os
import sys
import time

def process_image_keras_ocr(image_path, output_path, preprocess=False):
    """
    Process image with Keras OCR
    
    Args:
        image_path: Path to input image
        output_path: Path to save results (optional)
        preprocess: Apply image preprocessing (default: False)
    
    Returns:
        dict: OCR results with text, confidence, processing_time
    """
    start_time = time.time()
    
    try:
        # Import keras_ocr
        import keras_ocr
        
        # Initialize pipeline
        pipeline = keras_ocr.pipeline.Pipeline()
        
        # Read image
        image = keras_ocr.tools.read(image_path)
        
        # Apply preprocessing if requested
        if preprocess:
            # Enhanced preprocessing for book covers
            # 1. Increase contrast using CLAHE on LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            # Merge back
            lab = cv2.merge([l, a, b])
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # 2. Apply sharpening
            kernel_sharpening = np.array([[-1,-1,-1],
                                          [-1, 9,-1],
                                          [-1,-1,-1]])
            image = cv2.filter2D(image, -1, kernel_sharpening)
            
            # 3. Denoise slightly
            image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        
        # Perform OCR
        prediction_groups = pipeline.recognize([image])
        predictions = prediction_groups[0]
        
        # Extract text
        text_parts = []
        confidences = []
        simple_predictions = []
        
        for word, box in predictions:
            text_parts.append(word)
            # Keras OCR doesn't provide confidence directly
            # Use a default high confidence for detected text
            confidences.append(0.95)
            # Convert numpy array to list for JSON serialization
            simple_predictions.append({
                'word': word,
                'bbox': box.tolist() if hasattr(box, 'tolist') else box
            })
        
        full_text = ' '.join(text_parts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        processing_time = time.time() - start_time
        
        result = {
            'success': True,
            'text': full_text,
            'confidence': avg_confidence,
            'word_count': len(text_parts),
            'processing_time': processing_time,
            'engine': 'Keras OCR',
            'details': simple_predictions
        }
        
        # Save results if output path provided
        if output_path:
            # Save as JSON
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        
        return result
        
    except ImportError as e:
        return {
            'success': False,
            'error': f'Keras OCR not installed: {str(e)}',
            'text': '',
            'confidence': 0,
            'word_count': 0,
            'processing_time': time.time() - start_time,
            'engine': 'Keras OCR'
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'text': '',
            'confidence': 0,
            'word_count': 0,
            'processing_time': time.time() - start_time,
            'engine': 'Keras OCR'
        }


# Test function
if __name__ == "__main__":
    # Set UTF-8 encoding for console
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='Keras OCR Standalone')
    parser.add_argument('image_path', help='Path to input image')
    parser.add_argument('output_path', nargs='?', help='Path to save JSON results')
    parser.add_argument('--preprocess', action='store_true', help='Apply image preprocessing')
    
    args = parser.parse_args()
    
    print(f"Testing Keras OCR with: {args.image_path}")
    if args.preprocess:
        print("🔧 Preprocessing: ENABLED")
    result = process_image_keras_ocr(args.image_path, args.output_path, args.preprocess)
    
    # Print to console (for debugging)
    if result['success']:
        print("SUCCESS!")
        print(f"Text: {result['text'][:200] if result['text'] else 'No text detected'}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Time: {result['processing_time']:.2f}s")
    else:
        print(f"FAILED: {result['error']}")
    
    # Always save JSON if output_path provided (even on failure)
    if args.output_path:
        import json
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
