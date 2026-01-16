/*
 * EE 4065 - Embedded Digital Image Processing
 * Question 3: Upsampling and Downsampling on ESP32-CAM
 * 
 * Student: KAAN ATALAY
 * ID: 150720057
 * 
 * This code implements upsampling and downsampling operations
 * with support for non-integer scale factors (e.g., 1.5, 2/3).
 */

#include "esp_camera.h"
#include "Arduino.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"
#include <math.h>

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
#define INPUT_WIDTH   96
#define INPUT_HEIGHT  96
#define MAX_OUTPUT_SIZE (640 * 480)  // Maximum output buffer size

// ==================== INTERPOLATION METHODS ====================
typedef enum {
    INTERP_NEAREST,     // Nearest neighbor
    INTERP_BILINEAR,    // Bilinear interpolation
    INTERP_BICUBIC      // Bicubic interpolation (simplified)
} InterpolationMethod;

// ==================== GLOBAL BUFFERS ====================
uint8_t* input_buffer = NULL;
uint8_t* output_buffer = NULL;
int output_width, output_height;

// ==================== FUNCTION PROTOTYPES ====================
void initCamera();
bool allocateBuffers();
bool captureGrayscale();
void upsample(float scale_x, float scale_y, InterpolationMethod method);
void downsample(float scale_x, float scale_y, InterpolationMethod method);
void resizeImage(float scale_x, float scale_y, InterpolationMethod method);
uint8_t nearestNeighbor(float x, float y, int src_width, int src_height, uint8_t* src);
uint8_t bilinearInterpolate(float x, float y, int src_width, int src_height, uint8_t* src);
uint8_t bicubicInterpolate(float x, float y, int src_width, int src_height, uint8_t* src);
float cubicWeight(float t);
void sendImageSerial(uint8_t* data, int width, int height);
void printImageStats(uint8_t* data, int width, int height, const char* name);

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
    config.pixel_format = PIXFORMAT_GRAYSCALE;
    config.frame_size = FRAMESIZE_96X96;  // 96x96 grayscale
    config.jpeg_quality = 12;
    config.fb_count = 1;
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Camera init failed: 0x%x\n", err);
        return;
    }
    Serial.println("Camera initialized!");
}

// ==================== BUFFER ALLOCATION ====================
bool allocateBuffers() {
    input_buffer = (uint8_t*)ps_malloc(INPUT_WIDTH * INPUT_HEIGHT);
    output_buffer = (uint8_t*)ps_malloc(MAX_OUTPUT_SIZE);
    
    if (!input_buffer || !output_buffer) {
        Serial.println("Buffer allocation failed!");
        return false;
    }
    
    Serial.println("Buffers allocated in PSRAM!");
    return true;
}

// ==================== CAPTURE GRAYSCALE IMAGE ====================
bool captureGrayscale() {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("Camera capture failed!");
        return false;
    }
    
    memcpy(input_buffer, fb->buf, INPUT_WIDTH * INPUT_HEIGHT);
    esp_camera_fb_return(fb);
    return true;
}

// ==================== NEAREST NEIGHBOR INTERPOLATION ====================
uint8_t nearestNeighbor(float x, float y, int src_width, int src_height, uint8_t* src) {
    int ix = (int)(x + 0.5f);
    int iy = (int)(y + 0.5f);
    
    // Clamp to valid range
    ix = max(0, min(ix, src_width - 1));
    iy = max(0, min(iy, src_height - 1));
    
    return src[iy * src_width + ix];
}

// ==================== BILINEAR INTERPOLATION ====================
uint8_t bilinearInterpolate(float x, float y, int src_width, int src_height, uint8_t* src) {
    // Get integer and fractional parts
    int x0 = (int)floor(x);
    int y0 = (int)floor(y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    
    // Clamp to valid range
    x0 = max(0, min(x0, src_width - 1));
    x1 = max(0, min(x1, src_width - 1));
    y0 = max(0, min(y0, src_height - 1));
    y1 = max(0, min(y1, src_height - 1));
    
    // Fractional parts
    float fx = x - floor(x);
    float fy = y - floor(y);
    
    // Get four neighboring pixels
    float p00 = src[y0 * src_width + x0];
    float p01 = src[y0 * src_width + x1];
    float p10 = src[y1 * src_width + x0];
    float p11 = src[y1 * src_width + x1];
    
    // Bilinear interpolation formula
    float result = p00 * (1 - fx) * (1 - fy) +
                   p01 * fx * (1 - fy) +
                   p10 * (1 - fx) * fy +
                   p11 * fx * fy;
    
    return (uint8_t)max(0.0f, min(255.0f, result));
}

// ==================== CUBIC WEIGHT FUNCTION ====================
float cubicWeight(float t) {
    // Catmull-Rom spline weight function
    t = fabs(t);
    if (t < 1.0f) {
        return 1.5f * t * t * t - 2.5f * t * t + 1.0f;
    } else if (t < 2.0f) {
        return -0.5f * t * t * t + 2.5f * t * t - 4.0f * t + 2.0f;
    }
    return 0.0f;
}

// ==================== BICUBIC INTERPOLATION ====================
uint8_t bicubicInterpolate(float x, float y, int src_width, int src_height, uint8_t* src) {
    int x0 = (int)floor(x);
    int y0 = (int)floor(y);
    
    float fx = x - x0;
    float fy = y - y0;
    
    float result = 0.0f;
    float weight_sum = 0.0f;
    
    // 4x4 kernel
    for (int j = -1; j <= 2; j++) {
        for (int i = -1; i <= 2; i++) {
            int px = max(0, min(x0 + i, src_width - 1));
            int py = max(0, min(y0 + j, src_height - 1));
            
            float weight = cubicWeight(i - fx) * cubicWeight(j - fy);
            result += src[py * src_width + px] * weight;
            weight_sum += weight;
        }
    }
    
    if (weight_sum > 0) {
        result /= weight_sum;
    }
    
    return (uint8_t)max(0.0f, min(255.0f, result));
}

// ==================== MAIN RESIZE FUNCTION ====================
/*
 * Unified resize function that handles both upsampling and downsampling
 * with non-integer scale factors.
 * 
 * Parameters:
 *   scale_x: Horizontal scale factor (e.g., 1.5, 0.667)
 *   scale_y: Vertical scale factor
 *   method: Interpolation method to use
 */
void resizeImage(float scale_x, float scale_y, InterpolationMethod method) {
    // Calculate output dimensions
    output_width = (int)(INPUT_WIDTH * scale_x + 0.5f);
    output_height = (int)(INPUT_HEIGHT * scale_y + 0.5f);
    
    // Safety check
    if (output_width * output_height > MAX_OUTPUT_SIZE) {
        Serial.println("Output size exceeds buffer! Limiting...");
        float limit_scale = sqrt((float)MAX_OUTPUT_SIZE / (output_width * output_height));
        output_width = (int)(output_width * limit_scale);
        output_height = (int)(output_height * limit_scale);
    }
    
    Serial.printf("Resizing: %dx%d -> %dx%d (scale: %.3f x %.3f)\n",
                  INPUT_WIDTH, INPUT_HEIGHT, output_width, output_height, scale_x, scale_y);
    
    unsigned long start_time = millis();
    
    // Perform interpolation
    for (int y = 0; y < output_height; y++) {
        for (int x = 0; x < output_width; x++) {
            // Map output coordinates to input coordinates
            float src_x = (x + 0.5f) / scale_x - 0.5f;
            float src_y = (y + 0.5f) / scale_y - 0.5f;
            
            // Clamp to valid range
            src_x = max(0.0f, min(src_x, (float)(INPUT_WIDTH - 1)));
            src_y = max(0.0f, min(src_y, (float)(INPUT_HEIGHT - 1)));
            
            uint8_t pixel;
            switch (method) {
                case INTERP_NEAREST:
                    pixel = nearestNeighbor(src_x, src_y, INPUT_WIDTH, INPUT_HEIGHT, input_buffer);
                    break;
                case INTERP_BILINEAR:
                    pixel = bilinearInterpolate(src_x, src_y, INPUT_WIDTH, INPUT_HEIGHT, input_buffer);
                    break;
                case INTERP_BICUBIC:
                    pixel = bicubicInterpolate(src_x, src_y, INPUT_WIDTH, INPUT_HEIGHT, input_buffer);
                    break;
                default:
                    pixel = nearestNeighbor(src_x, src_y, INPUT_WIDTH, INPUT_HEIGHT, input_buffer);
            }
            
            output_buffer[y * output_width + x] = pixel;
        }
    }
    
    unsigned long elapsed = millis() - start_time;
    Serial.printf("Resize completed in %lu ms\n", elapsed);
}

// ==================== UPSAMPLING WRAPPER ====================
void upsample(float scale_x, float scale_y, InterpolationMethod method) {
    Serial.println("\n========== UPSAMPLING ==========");
    
    if (scale_x <= 1.0f && scale_y <= 1.0f) {
        Serial.println("Warning: Scale factors <= 1.0, this is downsampling!");
    }
    
    resizeImage(scale_x, scale_y, method);
    
    Serial.printf("Upsampled from %dx%d to %dx%d\n",
                  INPUT_WIDTH, INPUT_HEIGHT, output_width, output_height);
}

// ==================== DOWNSAMPLING WRAPPER ====================
void downsample(float scale_x, float scale_y, InterpolationMethod method) {
    Serial.println("\n========== DOWNSAMPLING ==========");
    
    if (scale_x >= 1.0f && scale_y >= 1.0f) {
        Serial.println("Warning: Scale factors >= 1.0, this is upsampling!");
    }
    
    resizeImage(scale_x, scale_y, method);
    
    Serial.printf("Downsampled from %dx%d to %dx%d\n",
                  INPUT_WIDTH, INPUT_HEIGHT, output_width, output_height);
}

// ==================== ANTI-ALIASING DOWNSAMPLE ====================
/*
 * For downsampling, apply a low-pass filter before sampling
 * to prevent aliasing artifacts.
 */
void downsampleWithAntiAlias(float scale_x, float scale_y) {
    Serial.println("\n========== DOWNSAMPLING WITH ANTI-ALIASING ==========");
    
    // Calculate filter kernel size based on scale factor
    int kernel_x = (int)(1.0f / scale_x) + 1;
    int kernel_y = (int)(1.0f / scale_y) + 1;
    kernel_x = min(kernel_x, 7);  // Limit kernel size
    kernel_y = min(kernel_y, 7);
    
    Serial.printf("Using %dx%d averaging kernel\n", kernel_x, kernel_y);
    
    // Apply box filter to input (in-place would need temp buffer)
    // For simplicity, we'll use area averaging during sampling
    
    output_width = (int)(INPUT_WIDTH * scale_x + 0.5f);
    output_height = (int)(INPUT_HEIGHT * scale_y + 0.5f);
    
    unsigned long start_time = millis();
    
    for (int y = 0; y < output_height; y++) {
        for (int x = 0; x < output_width; x++) {
            // Calculate source region
            float src_x0 = x / scale_x;
            float src_y0 = y / scale_y;
            float src_x1 = (x + 1) / scale_x;
            float src_y1 = (y + 1) / scale_y;
            
            // Area averaging
            float sum = 0;
            int count = 0;
            
            int ix0 = (int)floor(src_x0);
            int iy0 = (int)floor(src_y0);
            int ix1 = (int)ceil(src_x1);
            int iy1 = (int)ceil(src_y1);
            
            for (int sy = iy0; sy < iy1; sy++) {
                for (int sx = ix0; sx < ix1; sx++) {
                    int px = max(0, min(sx, INPUT_WIDTH - 1));
                    int py = max(0, min(sy, INPUT_HEIGHT - 1));
                    sum += input_buffer[py * INPUT_WIDTH + px];
                    count++;
                }
            }
            
            output_buffer[y * output_width + x] = (uint8_t)(sum / count);
        }
    }
    
    unsigned long elapsed = millis() - start_time;
    Serial.printf("Anti-aliased downsample completed in %lu ms\n", elapsed);
}

// ==================== PRINT IMAGE STATISTICS ====================
void printImageStats(uint8_t* data, int width, int height, const char* name) {
    uint32_t sum = 0;
    uint8_t min_val = 255, max_val = 0;
    
    int size = width * height;
    for (int i = 0; i < size; i++) {
        sum += data[i];
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }
    
    float mean = (float)sum / size;
    
    // Calculate variance
    float var_sum = 0;
    for (int i = 0; i < size; i++) {
        float diff = data[i] - mean;
        var_sum += diff * diff;
    }
    float std_dev = sqrt(var_sum / size);
    
    Serial.printf("[%s Stats]\n", name);
    Serial.printf("  Size: %dx%d (%d pixels)\n", width, height, size);
    Serial.printf("  Min: %d, Max: %d\n", min_val, max_val);
    Serial.printf("  Mean: %.2f, StdDev: %.2f\n", mean, std_dev);
}

// ==================== SEND IMAGE VIA SERIAL ====================
void sendImageSerial(uint8_t* data, int width, int height) {
    // Send header
    Serial.write(0xFF);
    Serial.write(0xAA);
    Serial.write((width >> 8) & 0xFF);
    Serial.write(width & 0xFF);
    Serial.write((height >> 8) & 0xFF);
    Serial.write(height & 0xFF);
    
    // Send data
    Serial.write(data, width * height);
    
    // Send footer
    Serial.write(0xFF);
    Serial.write(0x55);
}

// ==================== SETUP ====================
void setup() {
    WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
    
    Serial.begin(115200);
    delay(1000);
    
    Serial.println("\n========================================");
    Serial.println("EE 4065 - Upsampling/Downsampling");
    Serial.println("Student: KAAN ATALAY (150720057)");
    Serial.println("========================================\n");
    
    initCamera();
    
    if (!allocateBuffers()) {
        while (1) delay(1000);
    }
}

// ==================== MAIN LOOP ====================
void loop() {
    Serial.println("\n=== New Processing Cycle ===");
    
    // Capture image
    if (!captureGrayscale()) {
        delay(1000);
        return;
    }
    Serial.println("Image captured!");
    printImageStats(input_buffer, INPUT_WIDTH, INPUT_HEIGHT, "Original");
    
    // ============ TEST UPSAMPLING ============
    
    // Test 1: Integer upsampling (2x)
    Serial.println("\n--- Test 1: 2x Upsampling (Bilinear) ---");
    upsample(2.0f, 2.0f, INTERP_BILINEAR);
    printImageStats(output_buffer, output_width, output_height, "Upsampled 2x");
    
    // Test 2: Non-integer upsampling (1.5x)
    Serial.println("\n--- Test 2: 1.5x Upsampling (Bilinear) ---");
    upsample(1.5f, 1.5f, INTERP_BILINEAR);
    printImageStats(output_buffer, output_width, output_height, "Upsampled 1.5x");
    
    // ============ TEST DOWNSAMPLING ============
    
    // Test 3: Integer downsampling (0.5x = 1/2)
    Serial.println("\n--- Test 3: 0.5x Downsampling (Bilinear) ---");
    downsample(0.5f, 0.5f, INTERP_BILINEAR);
    printImageStats(output_buffer, output_width, output_height, "Downsampled 0.5x");
    
    // Test 4: Non-integer downsampling (2/3 = 0.667x)
    Serial.println("\n--- Test 4: 2/3 Downsampling (Bilinear) ---");
    downsample(2.0f/3.0f, 2.0f/3.0f, INTERP_BILINEAR);
    printImageStats(output_buffer, output_width, output_height, "Downsampled 2/3");
    
    // Test 5: Anti-aliased downsampling
    Serial.println("\n--- Test 5: 0.5x Downsampling with Anti-Aliasing ---");
    downsampleWithAntiAlias(0.5f, 0.5f);
    printImageStats(output_buffer, output_width, output_height, "Anti-aliased 0.5x");
    
    // ============ COMPARE INTERPOLATION METHODS ============
    Serial.println("\n\n===== INTERPOLATION METHOD COMPARISON =====");
    
    Serial.println("\n--- Nearest Neighbor (1.5x) ---");
    upsample(1.5f, 1.5f, INTERP_NEAREST);
    printImageStats(output_buffer, output_width, output_height, "Nearest 1.5x");
    
    Serial.println("\n--- Bilinear (1.5x) ---");
    upsample(1.5f, 1.5f, INTERP_BILINEAR);
    printImageStats(output_buffer, output_width, output_height, "Bilinear 1.5x");
    
    Serial.println("\n--- Bicubic (1.5x) ---");
    upsample(1.5f, 1.5f, INTERP_BICUBIC);
    printImageStats(output_buffer, output_width, output_height, "Bicubic 1.5x");
    
    Serial.println("\n=== Processing Complete ===\n");
    
    delay(10000);  // Wait before next cycle
}
