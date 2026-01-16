"""
EE 4065 - Embedded Digital Image Processing
Question 1a: Size-Based Thresholding on PC (Python)

Student: KAAN ATALAY
ID: 150720057

This script performs thresholding on an image captured from ESP32-CAM.
The threshold is automatically determined to extract an object with approximately 1000 pixels.
"""

import numpy as np
import cv2
import serial
import struct
import time

# ==================== CONFIGURATION ====================
TARGET_OBJECT_SIZE = 1000  # Target number of pixels for the object
TOLERANCE = 50  # Acceptable deviation from target size
SERIAL_PORT = '/dev/tty.usbserial-0001'  # Change according to your system
BAUD_RATE = 115200
IMAGE_WIDTH = 96
IMAGE_HEIGHT = 96

# ==================== HELPER FUNCTIONS ====================

def find_optimal_threshold(gray_image, target_size=TARGET_OBJECT_SIZE, tolerance=TOLERANCE):
    """
    Find the optimal threshold value that extracts an object with the target pixel count.
    Uses binary search for efficiency.
    
    Parameters:
        gray_image: Grayscale input image
        target_size: Desired number of white pixels after thresholding
        tolerance: Acceptable deviation from target size
    
    Returns:
        optimal_threshold: The threshold value that best matches target size
        binary_image: The resulting binary image
        actual_size: The actual object size achieved
    """
    low, high = 0, 255
    best_threshold = 128
    best_diff = float('inf')
    best_binary = None
    best_size = 0
    
    # Binary search for optimal threshold
    while low <= high:
        mid = (low + high) // 2
        
        # Apply threshold
        _, binary = cv2.threshold(gray_image, mid, 255, cv2.THRESH_BINARY)
        
        # Count white pixels (object pixels)
        white_pixels = np.sum(binary == 255)
        diff = abs(white_pixels - target_size)
        
        # Update best if this is closer to target
        if diff < best_diff:
            best_diff = diff
            best_threshold = mid
            best_binary = binary.copy()
            best_size = white_pixels
        
        # Check if within tolerance
        if diff <= tolerance:
            return mid, binary, white_pixels
        
        # Adjust search range
        # If too many white pixels, increase threshold (make it darker)
        # If too few white pixels, decrease threshold (make it brighter)
        if white_pixels > target_size:
            low = mid + 1
        else:
            high = mid - 1
    
    return best_threshold, best_binary, best_size


def histogram_based_threshold(gray_image, target_size=TARGET_OBJECT_SIZE):
    """
    Alternative method: Use histogram analysis to find threshold.
    Assumes the brightest 'target_size' pixels belong to the object.
    
    Parameters:
        gray_image: Grayscale input image
        target_size: Desired number of object pixels
    
    Returns:
        threshold: Computed threshold value
        binary_image: Resulting binary image
    """
    # Calculate histogram
    hist, bins = np.histogram(gray_image.flatten(), bins=256, range=(0, 256))
    
    # Find threshold from the bright end
    cumsum = np.cumsum(hist[::-1])  # Cumulative sum from bright to dark
    
    # Find the index where cumsum exceeds target_size
    threshold_idx = np.argmax(cumsum >= target_size)
    threshold = 255 - threshold_idx
    
    # Apply threshold
    _, binary = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    
    return threshold, binary


def receive_image_from_esp32(ser):
    """
    Receive image data from ESP32-CAM via serial.
    
    Parameters:
        ser: Serial port object
    
    Returns:
        image: Received grayscale image as numpy array
    """
    # Wait for start marker
    while True:
        if ser.read(1) == b'\xFF':
            if ser.read(1) == b'\xD8':  # JPEG start marker
                break
    
    # Read image data
    data = b'\xFF\xD8'
    while True:
        byte = ser.read(1)
        data += byte
        if len(data) >= 2 and data[-2:] == b'\xFF\xD9':  # JPEG end marker
            break
    
    # Decode JPEG
    nparr = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    return image


def process_and_display(image, target_size=TARGET_OBJECT_SIZE):
    """
    Process image and display results.
    
    Parameters:
        image: Input grayscale image
        target_size: Target object size in pixels
    """
    print(f"Image shape: {image.shape}")
    print(f"Target object size: {target_size} pixels")
    
    # Method 1: Binary Search
    threshold1, binary1, size1 = find_optimal_threshold(image, target_size)
    print(f"\n[Binary Search Method]")
    print(f"  Optimal threshold: {threshold1}")
    print(f"  Object size: {size1} pixels")
    print(f"  Difference from target: {abs(size1 - target_size)} pixels")
    
    # Method 2: Histogram-based
    threshold2, binary2 = histogram_based_threshold(image, target_size)
    size2 = np.sum(binary2 == 255)
    print(f"\n[Histogram Method]")
    print(f"  Threshold: {threshold2}")
    print(f"  Object size: {size2} pixels")
    print(f"  Difference from target: {abs(size2 - target_size)} pixels")
    
    # Find connected components for validation
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary1, connectivity=8)
    print(f"\n[Connected Components Analysis]")
    print(f"  Number of objects found: {num_labels - 1}")  # Exclude background
    
    # Find the component closest to target size
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if abs(area - target_size) < target_size * 0.5:  # Within 50% of target
            print(f"  Object {i}: Area = {area} pixels, Centroid = {centroids[i]}")
    
    # Display results
    cv2.imshow('Original', image)
    cv2.imshow('Binary (Search)', binary1)
    cv2.imshow('Binary (Histogram)', binary2)
    
    # Create visualization with object highlighted
    result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(binary1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    cv2.imshow('Detected Object', result)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return threshold1, binary1


def main():
    """Main function to run the thresholding operation."""
    print("=" * 60)
    print("EE 4065 - Size-Based Thresholding")
    print("Student: KAAN ATALAY (150720057)")
    print("=" * 60)
    
    # Option 1: Load from file (for testing)
    test_mode = True  # Set to False for ESP32-CAM mode
    
    if test_mode:
        # Create a test image with a bright object
        print("\n[TEST MODE] Creating synthetic test image...")
        test_image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
        
        # Add random background noise (dark)
        test_image = np.random.randint(20, 80, (IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
        
        # Add a bright object (approximately 1000 pixels)
        # A circle with radius ~18 pixels has area ≈ π*18² ≈ 1018 pixels
        # For 96x96 image, adjust radius to fit
        center = (IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2)
        cv2.circle(test_image, center, 18, 200, -1)  # Bright filled circle (~1000 pixels)
        
        # Add some noise to the object
        object_mask = np.zeros_like(test_image)
        cv2.circle(object_mask, center, 18, 255, -1)
        noise = np.random.randint(-20, 20, test_image.shape, dtype=np.int16)
        test_image = np.clip(test_image.astype(np.int16) + noise * (object_mask > 0).astype(np.int16) // 2, 0, 255).astype(np.uint8)
        
        image = test_image
        
    else:
        # Option 2: Receive from ESP32-CAM
        print("\n[ESP32-CAM MODE] Connecting to ESP32-CAM...")
        try:
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=10)
            time.sleep(2)  # Wait for connection
            print("Connected! Waiting for image...")
            
            image = receive_image_from_esp32(ser)
            ser.close()
            
            if image is None:
                print("Error: Failed to receive image!")
                return
                
        except Exception as e:
            print(f"Error connecting to ESP32-CAM: {e}")
            print("Falling back to test mode...")
            test_mode = True
            # Create test image as fallback
            image = np.random.randint(20, 80, (IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
            cv2.circle(image, (IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2), 18, 200, -1)
    
    # Process the image
    threshold, binary = process_and_display(image, TARGET_OBJECT_SIZE)
    
    # Save results
    cv2.imwrite('original.png', image)
    cv2.imwrite('binary_result.png', binary)
    print(f"\nResults saved to 'original.png' and 'binary_result.png'")
    print(f"Final threshold value: {threshold}")


if __name__ == "__main__":
    main()
