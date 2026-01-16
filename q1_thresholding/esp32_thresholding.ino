/*
 * EE 4065 - Embedded Digital Image Processing
 * Question 1b: Size-Based Thresholding on ESP32-CAM (C Code)
 * 
 * Student: KAAN ATALAY
 * ID: 150720057
 * 
 * This code performs automatic thresholding on ESP32-CAM to extract
 * a bright object with approximately 1000 pixels from a dark background.
 */

#include "esp_camera.h"
#include "esp_timer.h"
#include "img_converters.h"
#include "Arduino.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

// ==================== CAMERA PIN DEFINITIONS (AI-THINKER) ====================
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// ==================== CONFIGURATION ====================
#define TARGET_OBJECT_SIZE  1000    // Target number of pixels for object
#define SIZE_TOLERANCE      50      // Acceptable deviation from target
#define IMAGE_WIDTH         96
#define IMAGE_HEIGHT        96
#define GRAYSCALE_SIZE      (IMAGE_WIDTH * IMAGE_HEIGHT)

// ==================== GLOBAL VARIABLES ====================
uint8_t grayscale_buffer[GRAYSCALE_SIZE];
uint8_t binary_buffer[GRAYSCALE_SIZE];
uint8_t optimal_threshold = 128;

// ==================== FUNCTION PROTOTYPES ====================
void initCamera();
bool captureAndConvertToGrayscale();
uint32_t countWhitePixels(uint8_t threshold);
uint8_t findOptimalThreshold();
uint8_t histogramBasedThreshold();
void applyThreshold(uint8_t threshold);
void sendResultsToSerial();
void analyzeConnectedComponents();

// ==================== CAMERA INITIALIZATION ====================
void initCamera() {
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sscb_sda = SIOD_GPIO_NUM;
    config.pin_sscb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_GRAYSCALE;  // Direct grayscale capture
    config.frame_size = FRAMESIZE_96X96;        // 96x96 grayscale
    config.jpeg_quality = 12;
    config.fb_count = 1;
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;

    // Initialize camera
    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Camera init failed with error 0x%x\n", err);
        return;
    }
    
    // Camera settings for better contrast
    sensor_t *s = esp_camera_sensor_get();
    s->set_brightness(s, 0);
    s->set_contrast(s, 1);
    s->set_saturation(s, 0);
    s->set_gainceiling(s, (gainceiling_t)6);
    s->set_whitebal(s, 1);
    s->set_awb_gain(s, 1);
    s->set_exposure_ctrl(s, 1);
    s->set_aec2(s, 1);
    s->set_gain_ctrl(s, 1);
    s->set_agc_gain(s, 0);
    
    Serial.println("Camera initialized successfully!");
}

// ==================== CAPTURE AND CONVERT ====================
bool captureAndConvertToGrayscale() {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("Camera capture failed!");
        return false;
    }
    
    // Copy grayscale data (already grayscale from camera)
    if (fb->format == PIXFORMAT_GRAYSCALE) {
        memcpy(grayscale_buffer, fb->buf, GRAYSCALE_SIZE);
    } else {
        // If RGB565, convert to grayscale
        uint16_t *rgb_buf = (uint16_t*)fb->buf;
        for (int i = 0; i < GRAYSCALE_SIZE; i++) {
            uint16_t pixel = rgb_buf[i];
            uint8_t r = (pixel >> 11) & 0x1F;
            uint8_t g = (pixel >> 5) & 0x3F;
            uint8_t b = pixel & 0x1F;
            // Convert to 8-bit and calculate luminance
            r = (r * 255) / 31;
            g = (g * 255) / 63;
            b = (b * 255) / 31;
            grayscale_buffer[i] = (uint8_t)(0.299 * r + 0.587 * g + 0.114 * b);
        }
    }
    
    esp_camera_fb_return(fb);
    return true;
}

// ==================== COUNT WHITE PIXELS ====================
uint32_t countWhitePixels(uint8_t threshold) {
    uint32_t count = 0;
    for (int i = 0; i < GRAYSCALE_SIZE; i++) {
        if (grayscale_buffer[i] > threshold) {
            count++;
        }
    }
    return count;
}

// ==================== BINARY SEARCH FOR OPTIMAL THRESHOLD ====================
uint8_t findOptimalThreshold() {
    uint8_t low = 0, high = 255;
    uint8_t best_threshold = 128;
    int32_t best_diff = INT32_MAX;
    
    Serial.println("\n[Binary Search Thresholding]");
    Serial.printf("Target size: %d pixels\n", TARGET_OBJECT_SIZE);
    
    while (low <= high) {
        uint8_t mid = (low + high) / 2;
        uint32_t white_count = countWhitePixels(mid);
        int32_t diff = abs((int32_t)white_count - TARGET_OBJECT_SIZE);
        
        Serial.printf("  Threshold: %d -> White pixels: %lu (diff: %ld)\n", 
                      mid, white_count, diff);
        
        // Update best if closer to target
        if (diff < best_diff) {
            best_diff = diff;
            best_threshold = mid;
        }
        
        // Check if within tolerance
        if (diff <= SIZE_TOLERANCE) {
            Serial.printf("Found threshold within tolerance!\n");
            return mid;
        }
        
        // Adjust search range
        if (white_count > TARGET_OBJECT_SIZE) {
            low = mid + 1;  // Need higher threshold (fewer white pixels)
        } else {
            high = mid - 1; // Need lower threshold (more white pixels)
        }
        
        // Prevent infinite loop
        if (low > high) break;
    }
    
    Serial.printf("Best threshold found: %d\n", best_threshold);
    return best_threshold;
}

// ==================== HISTOGRAM-BASED THRESHOLD ====================
uint8_t histogramBasedThreshold() {
    // Build histogram
    uint32_t histogram[256] = {0};
    for (int i = 0; i < GRAYSCALE_SIZE; i++) {
        histogram[grayscale_buffer[i]]++;
    }
    
    // Find threshold from bright end (cumulative sum)
    uint32_t cumsum = 0;
    for (int i = 255; i >= 0; i--) {
        cumsum += histogram[i];
        if (cumsum >= TARGET_OBJECT_SIZE) {
            Serial.printf("[Histogram Method] Threshold: %d\n", i);
            return (uint8_t)i;
        }
    }
    
    return 128; // Default fallback
}

// ==================== APPLY THRESHOLD ====================
void applyThreshold(uint8_t threshold) {
    for (int i = 0; i < GRAYSCALE_SIZE; i++) {
        binary_buffer[i] = (grayscale_buffer[i] > threshold) ? 255 : 0;
    }
}

// ==================== SIMPLE CONNECTED COMPONENT ANALYSIS ====================
void analyzeConnectedComponents() {
    // Simple 8-connected component labeling (for demonstration)
    // Full implementation would use Union-Find
    
    uint32_t object_pixels = 0;
    int16_t sum_x = 0, sum_y = 0;
    int16_t min_x = IMAGE_WIDTH, max_x = 0;
    int16_t min_y = IMAGE_HEIGHT, max_y = 0;
    
    // Calculate centroid and bounding box of white pixels
    for (int y = 0; y < IMAGE_HEIGHT; y++) {
        for (int x = 0; x < IMAGE_WIDTH; x++) {
            int idx = y * IMAGE_WIDTH + x;
            if (binary_buffer[idx] == 255) {
                object_pixels++;
                sum_x += x;
                sum_y += y;
                if (x < min_x) min_x = x;
                if (x > max_x) max_x = x;
                if (y < min_y) min_y = y;
                if (y > max_y) max_y = y;
            }
        }
    }
    
    if (object_pixels > 0) {
        float centroid_x = (float)sum_x / object_pixels;
        float centroid_y = (float)sum_y / object_pixels;
        
        Serial.println("\n[Object Analysis]");
        Serial.printf("  Total object pixels: %lu\n", object_pixels);
        Serial.printf("  Centroid: (%.1f, %.1f)\n", centroid_x, centroid_y);
        Serial.printf("  Bounding box: (%d,%d) to (%d,%d)\n", min_x, min_y, max_x, max_y);
        Serial.printf("  Bounding box size: %dx%d\n", max_x - min_x + 1, max_y - min_y + 1);
    }
}

// ==================== SEND RESULTS VIA SERIAL ====================
void sendResultsToSerial() {
    // Send binary image via serial (for PC visualization)
    Serial.println("\n[Sending Binary Image via Serial]");
    
    // Send start marker
    Serial.write(0xFF);
    Serial.write(0xAA);
    
    // Send dimensions
    Serial.write((IMAGE_WIDTH >> 8) & 0xFF);
    Serial.write(IMAGE_WIDTH & 0xFF);
    Serial.write((IMAGE_HEIGHT >> 8) & 0xFF);
    Serial.write(IMAGE_HEIGHT & 0xFF);
    
    // Send binary data (compressed: 1 bit per pixel)
    for (int i = 0; i < GRAYSCALE_SIZE; i += 8) {
        uint8_t packed = 0;
        for (int j = 0; j < 8 && (i + j) < GRAYSCALE_SIZE; j++) {
            if (binary_buffer[i + j] == 255) {
                packed |= (1 << (7 - j));
            }
        }
        Serial.write(packed);
    }
    
    // Send end marker
    Serial.write(0xFF);
    Serial.write(0x55);
    
    Serial.println("Binary image sent!");
}

// ==================== SETUP ====================
void setup() {
    // Disable brownout detector
    WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
    
    Serial.begin(115200);
    Serial.setDebugOutput(true);
    delay(1000);
    
    Serial.println("\n========================================");
    Serial.println("EE 4065 - Size-Based Thresholding");
    Serial.println("Student: KAAN ATALAY (150720057)");
    Serial.println("========================================\n");
    
    // Initialize camera
    initCamera();
}

// ==================== MAIN LOOP ====================
void loop() {
    Serial.println("\n--- New Capture Cycle ---");
    
    // Capture image
    if (!captureAndConvertToGrayscale()) {
        delay(1000);
        return;
    }
    Serial.println("Image captured successfully!");
    
    // Method 1: Binary Search Threshold
    unsigned long start_time = millis();
    uint8_t threshold1 = findOptimalThreshold();
    unsigned long search_time = millis() - start_time;
    Serial.printf("Binary search time: %lu ms\n", search_time);
    
    // Method 2: Histogram-based Threshold
    start_time = millis();
    uint8_t threshold2 = histogramBasedThreshold();
    unsigned long hist_time = millis() - start_time;
    Serial.printf("Histogram method time: %lu ms\n", hist_time);
    
    // Use binary search result (more accurate)
    optimal_threshold = threshold1;
    
    // Apply threshold
    applyThreshold(optimal_threshold);
    
    // Count final result
    uint32_t final_count = 0;
    for (int i = 0; i < GRAYSCALE_SIZE; i++) {
        if (binary_buffer[i] == 255) final_count++;
    }
    
    Serial.println("\n========== FINAL RESULTS ==========");
    Serial.printf("Optimal Threshold: %d\n", optimal_threshold);
    Serial.printf("Object Size: %lu pixels\n", final_count);
    Serial.printf("Target Size: %d pixels\n", TARGET_OBJECT_SIZE);
    Serial.printf("Difference: %ld pixels\n", abs((int32_t)final_count - TARGET_OBJECT_SIZE));
    
    // Analyze connected components
    analyzeConnectedComponents();
    
    // Send results
    sendResultsToSerial();
    
    Serial.println("\n===================================\n");
    
    // Wait before next capture
    delay(5000);
}
