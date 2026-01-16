/*
 * EE 4065 - Embedded Digital Image Processing
 * Question 5 (BONUS): FOMO and SSD+MobileNet on ESP32-CAM
 * 
 * Student: KAAN ATALAY
 * ID: 150720057
 * 
 * This code implements both FOMO and SSD+MobileNet for digit detection
 * on ESP32-CAM using TensorFlow Lite Micro.
 */

#include "esp_camera.h"
#include "Arduino.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"
#include <math.h>

// TensorFlow Lite Micro
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Include model headers
#include "fomo_model.h"
#include "ssd_model.h"

// ==================== CAMERA PINS (AI-THINKER) ====================
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
#define IMG_SIZE          96
#define FOMO_GRID_SIZE    12
#define NUM_CLASSES       10
#define CONF_THRESHOLD    0.5
#define TENSOR_ARENA_SIZE (250 * 1024)

// ==================== DETECTION STRUCTURE ====================
typedef struct {
    int class_id;
    float confidence;
    float x, y;       // Center position (normalized)
    float w, h;       // Size (normalized, for SSD)
} Detection;

#define MAX_DETECTIONS 20
Detection detections[MAX_DETECTIONS];
int num_detections = 0;

// ==================== GLOBAL VARIABLES ====================
uint8_t* tensor_arena = nullptr;
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;

const char* class_names[] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};

// Image buffers
uint8_t image_buffer[IMG_SIZE * IMG_SIZE];      // Grayscale
float input_data[IMG_SIZE * IMG_SIZE * 3];      // RGB (gray copied to 3 channels)

// ==================== FUNCTION PROTOTYPES ====================
void initCamera();
bool captureImage();
void preprocessImage();
void runFOMO();
void runSSD();
void postprocessFOMO(float* output, int grid_size);
void postprocessSSD(float* output, int num_anchors);
void printDetections(const char* model_name);
void nonMaxSuppression();

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
    config.pixel_format = PIXFORMAT_GRAYSCALE;  // Grayscale capture
    config.frame_size = FRAMESIZE_96X96;       // 96x96 grayscale
    config.jpeg_quality = 12;
    config.fb_count = 1;
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Camera init failed: 0x%x\n", err);
    }
    Serial.println("Camera initialized!");
}

// ==================== IMAGE CAPTURE ====================
bool captureImage() {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("Capture failed!");
        return false;
    }
    memcpy(image_buffer, fb->buf, fb->len);
    esp_camera_fb_return(fb);
    return true;
}

// ==================== PREPROCESSING ====================
void preprocessImage() {
    /*
     * Convert grayscale to 3-channel (R=G=B=gray)
     * and normalize to [0, 1]
     */
    int idx = 0;
    
    for (int y = 0; y < IMG_SIZE; y++) {
        for (int x = 0; x < IMG_SIZE; x++) {
            float gray = image_buffer[y * IMG_SIZE + x] / 255.0f;
            
            // Copy grayscale to all 3 channels
            input_data[idx++] = gray;
            input_data[idx++] = gray;
            input_data[idx++] = gray;
        }
    }
}

// ==================== FOMO INFERENCE ====================
void runFOMO() {
    Serial.println("\n--- Running FOMO ---");
    
    // Load model
    const tflite::Model* model = tflite::GetModel(fomo_model_tflite);
    if (!model) {
        Serial.println("Failed to load FOMO model!");
        return;
    }
    
    // Allocate arena
    if (!tensor_arena) {
        tensor_arena = (uint8_t*)ps_malloc(TENSOR_ARENA_SIZE);
    }
    
    static tflite::AllOpsResolver resolver;
    tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, 
                                          TENSOR_ARENA_SIZE, error_reporter);
    
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        Serial.println("Failed to allocate FOMO tensors!");
        return;
    }
    
    // Copy input
    float* input = interpreter.input(0)->data.f;
    memcpy(input, input_data, IMG_SIZE * IMG_SIZE * 3 * sizeof(float));
    
    // Run inference
    unsigned long start = millis();
    if (interpreter.Invoke() != kTfLiteOk) {
        Serial.println("FOMO inference failed!");
        return;
    }
    Serial.printf("FOMO inference: %lu ms\n", millis() - start);
    
    // Process output
    float* output = interpreter.output(0)->data.f;
    postprocessFOMO(output, FOMO_GRID_SIZE);
    
    printDetections("FOMO");
}

// ==================== FOMO POSTPROCESSING ====================
void postprocessFOMO(float* output, int grid_size) {
    /*
     * FOMO output: (grid_size, grid_size, num_classes + 1)
     * Channel 0 is background, channels 1-10 are digit classes
     */
    num_detections = 0;
    
    for (int y = 0; y < grid_size && num_detections < MAX_DETECTIONS; y++) {
        for (int x = 0; x < grid_size && num_detections < MAX_DETECTIONS; x++) {
            int base_idx = (y * grid_size + x) * (NUM_CLASSES + 1);
            
            // Find best class (skip background at index 0)
            int best_class = -1;
            float best_conf = CONF_THRESHOLD;
            
            for (int c = 1; c <= NUM_CLASSES; c++) {
                float conf = output[base_idx + c];
                if (conf > best_conf) {
                    best_conf = conf;
                    best_class = c - 1;  // Adjust for 0-based class index
                }
            }
            
            if (best_class >= 0) {
                detections[num_detections].class_id = best_class;
                detections[num_detections].confidence = best_conf;
                detections[num_detections].x = (x + 0.5f) / grid_size;
                detections[num_detections].y = (y + 0.5f) / grid_size;
                detections[num_detections].w = 1.0f / grid_size;
                detections[num_detections].h = 1.0f / grid_size;
                num_detections++;
            }
        }
    }
}

// ==================== SSD INFERENCE ====================
void runSSD() {
    Serial.println("\n--- Running SSD+MobileNet ---");
    
    const tflite::Model* model = tflite::GetModel(ssd_model_tflite);
    if (!model) {
        Serial.println("Failed to load SSD model!");
        return;
    }
    
    if (!tensor_arena) {
        tensor_arena = (uint8_t*)ps_malloc(TENSOR_ARENA_SIZE);
    }
    
    static tflite::AllOpsResolver resolver;
    tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                          TENSOR_ARENA_SIZE, error_reporter);
    
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        Serial.println("Failed to allocate SSD tensors!");
        return;
    }
    
    float* input = interpreter.input(0)->data.f;
    memcpy(input, input_data, IMG_SIZE * IMG_SIZE * 3 * sizeof(float));
    
    unsigned long start = millis();
    if (interpreter.Invoke() != kTfLiteOk) {
        Serial.println("SSD inference failed!");
        return;
    }
    Serial.printf("SSD inference: %lu ms\n", millis() - start);
    
    float* output = interpreter.output(0)->data.f;
    
    // Calculate number of anchors from output size
    TfLiteTensor* output_tensor = interpreter.output(0);
    int output_size = 1;
    for (int i = 0; i < output_tensor->dims->size; i++) {
        output_size *= output_tensor->dims->data[i];
    }
    
    // SSD output: [loc(4), conf(11)] per anchor
    int anchor_output_size = 4 + NUM_CLASSES + 1;
    int num_anchors = output_size / anchor_output_size;
    
    postprocessSSD(output, num_anchors);
    nonMaxSuppression();
    
    printDetections("SSD+MobileNet");
}

// ==================== SSD POSTPROCESSING ====================
void postprocessSSD(float* output, int num_anchors) {
    /*
     * SSD output per anchor: [dx, dy, dw, dh, bg, c0, c1, ..., c9]
     */
    num_detections = 0;
    
    // Default anchor parameters (simplified)
    float default_scale = 0.2f;
    
    for (int a = 0; a < num_anchors && num_detections < MAX_DETECTIONS; a++) {
        int base_idx = a * (4 + NUM_CLASSES + 1);
        
        // Get box predictions
        float dx = output[base_idx + 0];
        float dy = output[base_idx + 1];
        float dw = output[base_idx + 2];
        float dh = output[base_idx + 3];
        
        // Find best class (skip background at index 4)
        int best_class = -1;
        float best_conf = CONF_THRESHOLD;
        
        // Apply softmax to confidence scores
        float max_score = output[base_idx + 4];
        for (int c = 0; c <= NUM_CLASSES; c++) {
            if (output[base_idx + 4 + c] > max_score) {
                max_score = output[base_idx + 4 + c];
            }
        }
        
        float sum_exp = 0;
        float scores[NUM_CLASSES + 1];
        for (int c = 0; c <= NUM_CLASSES; c++) {
            scores[c] = exp(output[base_idx + 4 + c] - max_score);
            sum_exp += scores[c];
        }
        
        for (int c = 1; c <= NUM_CLASSES; c++) {
            float conf = scores[c] / sum_exp;
            if (conf > best_conf) {
                best_conf = conf;
                best_class = c - 1;
            }
        }
        
        if (best_class >= 0) {
            // Decode box (simplified anchor at grid center)
            int grid_idx = a % (24 * 24);  // Assuming first feature map
            int grid_x = grid_idx % 24;
            int grid_y = grid_idx / 24;
            
            float cx = (grid_x + 0.5f + dx * default_scale) / 24.0f;
            float cy = (grid_y + 0.5f + dy * default_scale) / 24.0f;
            float w = default_scale * exp(dw);
            float h = default_scale * exp(dh);
            
            detections[num_detections].class_id = best_class;
            detections[num_detections].confidence = best_conf;
            detections[num_detections].x = cx;
            detections[num_detections].y = cy;
            detections[num_detections].w = w;
            detections[num_detections].h = h;
            num_detections++;
        }
    }
}

// ==================== NMS ====================
void nonMaxSuppression() {
    // Simple NMS
    bool keep[MAX_DETECTIONS] = {true};
    
    for (int i = 0; i < num_detections; i++) {
        if (!keep[i]) continue;
        
        for (int j = i + 1; j < num_detections; j++) {
            if (!keep[j]) continue;
            
            if (detections[i].class_id == detections[j].class_id) {
                // Calculate IoU
                float xi1 = max(detections[i].x - detections[i].w/2,
                               detections[j].x - detections[j].w/2);
                float yi1 = max(detections[i].y - detections[i].h/2,
                               detections[j].y - detections[j].h/2);
                float xi2 = min(detections[i].x + detections[i].w/2,
                               detections[j].x + detections[j].w/2);
                float yi2 = min(detections[i].y + detections[i].h/2,
                               detections[j].y + detections[j].h/2);
                
                float inter = max(0.0f, xi2 - xi1) * max(0.0f, yi2 - yi1);
                float union_area = detections[i].w * detections[i].h +
                                   detections[j].w * detections[j].h - inter;
                
                if (inter / union_area > 0.4f) {
                    if (detections[i].confidence >= detections[j].confidence) {
                        keep[j] = false;
                    } else {
                        keep[i] = false;
                    }
                }
            }
        }
    }
    
    // Compact
    int write_idx = 0;
    for (int i = 0; i < num_detections; i++) {
        if (keep[i]) {
            if (write_idx != i) {
                detections[write_idx] = detections[i];
            }
            write_idx++;
        }
    }
    num_detections = write_idx;
}

// ==================== PRINT RESULTS ====================
void printDetections(const char* model_name) {
    Serial.printf("\n[%s Results]\n", model_name);
    
    if (num_detections == 0) {
        Serial.println("No digits detected.");
    } else {
        for (int i = 0; i < num_detections; i++) {
            Serial.printf("  Digit %s: %.1f%% at (%.2f, %.2f)\n",
                          class_names[detections[i].class_id],
                          detections[i].confidence * 100,
                          detections[i].x * IMG_SIZE,
                          detections[i].y * IMG_SIZE);
        }
    }
}

// ==================== SETUP ====================
void setup() {
    WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
    
    Serial.begin(115200);
    delay(1000);
    
    Serial.println("\n========================================");
    Serial.println("EE 4065 - FOMO & SSD+MobileNet Detection");
    Serial.println("Student: KAAN ATALAY (150720057)");
    Serial.println("========================================\n");
    
    initCamera();
}

// ==================== MAIN LOOP ====================
void loop() {
    Serial.println("\n=== Detection Cycle ===");
    
    if (!captureImage()) {
        delay(1000);
        return;
    }
    
    preprocessImage();
    
    // Run both models
    runFOMO();
    runSSD();
    
    Serial.println("\n===========================\n");
    
    delay(3000);
}
