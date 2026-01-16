/*
 * EE 4065 - Embedded Digital Image Processing
 * Question 2b: YOLO Handwritten Digit Detection on ESP32-CAM
 * 
 * Student: KAAN ATALAY
 * ID: 150720057
 * 
 * This code runs YOLO inference on ESP32-CAM for handwritten digit detection.
 * Uses TensorFlow Lite Micro for model inference.
 */

#include "esp_camera.h"
#include "esp_timer.h"
#include "Arduino.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

// TensorFlow Lite includes
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Include the trained model (converted to C array)
#include "digit_model.h"

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
#define MODEL_INPUT_WIDTH   96
#define MODEL_INPUT_HEIGHT  96
#define MODEL_INPUT_CHANNELS 3
#define NUM_CLASSES         10
#define CONFIDENCE_THRESHOLD 0.5
#define NMS_THRESHOLD       0.4

// TensorFlow Lite arena size
#define TENSOR_ARENA_SIZE   (300 * 1024)  // 300KB

// ==================== GLOBAL VARIABLES ====================
uint8_t tensor_arena[TENSOR_ARENA_SIZE];
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Detection results structure
struct Detection {
    int class_id;
    float confidence;
    float x, y, w, h;
};

Detection detections[100];
int num_detections = 0;

const char* class_names[] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};

// ==================== FUNCTION PROTOTYPES ====================
void initCamera();
void initTFLite();
bool captureImage(uint8_t* buffer);
void preprocessImage(uint8_t* src, int src_width, int src_height, float* dst);
void runInference();
void postprocessOutput();
void nonMaxSuppression();
void printResults();

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
        return;
    }
    
    Serial.println("Camera initialized!");
}

// ==================== TENSORFLOW LITE INITIALIZATION ====================
void initTFLite() {
    // Load model
    model = tflite::GetModel(digit_model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.printf("Model version mismatch: %d vs %d\n", 
                      model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }
    
    // Create resolver with all ops
    static tflite::AllOpsResolver resolver;
    
    // Build interpreter
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, TENSOR_ARENA_SIZE, error_reporter);
    interpreter = &static_interpreter;
    
    // Allocate tensors
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        Serial.println("AllocateTensors() failed!");
        return;
    }
    
    // Get input/output tensors
    input = interpreter->input(0);
    output = interpreter->output(0);
    
    Serial.println("TensorFlow Lite initialized!");
    Serial.printf("Input shape: [%d, %d, %d, %d]\n",
                  input->dims->data[0], input->dims->data[1],
                  input->dims->data[2], input->dims->data[3]);
    Serial.printf("Arena used: %d bytes\n", interpreter->arena_used_bytes());
}

// ==================== IMAGE CAPTURE ====================
bool captureImage(uint8_t* buffer) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("Camera capture failed!");
        return false;
    }
    
    memcpy(buffer, fb->buf, fb->len);
    esp_camera_fb_return(fb);
    return true;
}

// ==================== IMAGE PREPROCESSING ====================
void preprocessImage(uint8_t* src, int src_width, int src_height, float* dst) {
    /*
     * Preprocess grayscale image for YOLO inference:
     * 1. Grayscale to 3-channel (R=G=B=gray)
     * 2. Normalize to [0, 1]
     */
    
    int dst_idx = 0;
    
    for (int y = 0; y < MODEL_INPUT_HEIGHT; y++) {
        for (int x = 0; x < MODEL_INPUT_WIDTH; x++) {
            int src_idx = y * src_width + x;
            
            // Grayscale value
            float gray = src[src_idx] / 255.0f;
            
            // Copy to all 3 channels (R=G=B=gray)
            dst[dst_idx++] = gray;
            dst[dst_idx++] = gray;
            dst[dst_idx++] = gray;
        }
    }
}

// ==================== RUN INFERENCE ====================
void runInference() {
    unsigned long start_time = millis();
    
    TfLiteStatus invoke_status = interpreter->Invoke();
    
    if (invoke_status != kTfLiteOk) {
        Serial.println("Invoke failed!");
        return;
    }
    
    unsigned long inference_time = millis() - start_time;
    Serial.printf("Inference time: %lu ms\n", inference_time);
}

// ==================== YOLO OUTPUT POSTPROCESSING ====================
void postprocessOutput() {
    /*
     * YOLO output format (for each grid cell):
     * [x, y, w, h, objectness, class_0_prob, ..., class_9_prob]
     * 
     * For YOLOv8-nano, output is typically:
     * Shape: [1, num_predictions, 14] (4 bbox + 10 classes)
     */
    
    num_detections = 0;
    float* output_data = output->data.f;
    
    int num_predictions = output->dims->data[1];
    int prediction_size = output->dims->data[2];
    
    for (int i = 0; i < num_predictions && num_detections < 100; i++) {
        float* pred = output_data + i * prediction_size;
        
        // Get bounding box (normalized coordinates)
        float x = pred[0];
        float y = pred[1];
        float w = pred[2];
        float h = pred[3];
        
        // Find best class
        int best_class = 0;
        float best_score = 0;
        
        for (int c = 0; c < NUM_CLASSES; c++) {
            float score = pred[4 + c];
            if (score > best_score) {
                best_score = score;
                best_class = c;
            }
        }
        
        // Apply confidence threshold
        if (best_score >= CONFIDENCE_THRESHOLD) {
            detections[num_detections].class_id = best_class;
            detections[num_detections].confidence = best_score;
            detections[num_detections].x = x;
            detections[num_detections].y = y;
            detections[num_detections].w = w;
            detections[num_detections].h = h;
            num_detections++;
        }
    }
}

// ==================== NON-MAXIMUM SUPPRESSION ====================
float calculateIoU(Detection& a, Detection& b) {
    float x1 = max(a.x - a.w/2, b.x - b.w/2);
    float y1 = max(a.y - a.h/2, b.y - b.h/2);
    float x2 = min(a.x + a.w/2, b.x + b.w/2);
    float y2 = min(a.y + a.h/2, b.y + b.h/2);
    
    float intersection = max(0.0f, x2 - x1) * max(0.0f, y2 - y1);
    float union_area = a.w * a.h + b.w * b.h - intersection;
    
    return intersection / (union_area + 1e-6);
}

void nonMaxSuppression() {
    // Sort by confidence (simple bubble sort for small arrays)
    for (int i = 0; i < num_detections - 1; i++) {
        for (int j = i + 1; j < num_detections; j++) {
            if (detections[j].confidence > detections[i].confidence) {
                Detection temp = detections[i];
                detections[i] = detections[j];
                detections[j] = temp;
            }
        }
    }
    
    // Apply NMS
    bool keep[100] = {true};
    
    for (int i = 0; i < num_detections; i++) {
        if (!keep[i]) continue;
        
        for (int j = i + 1; j < num_detections; j++) {
            if (!keep[j]) continue;
            
            if (detections[i].class_id == detections[j].class_id) {
                float iou = calculateIoU(detections[i], detections[j]);
                if (iou > NMS_THRESHOLD) {
                    keep[j] = false;
                }
            }
        }
    }
    
    // Compact array
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
void printResults() {
    Serial.println("\n========== DETECTION RESULTS ==========");
    
    if (num_detections == 0) {
        Serial.println("No digits detected.");
    } else {
        for (int i = 0; i < num_detections; i++) {
            Serial.printf("Detection %d:\n", i + 1);
            Serial.printf("  Digit: %s\n", class_names[detections[i].class_id]);
            Serial.printf("  Confidence: %.2f%%\n", detections[i].confidence * 100);
            Serial.printf("  Position: (%.2f, %.2f)\n", 
                          detections[i].x * MODEL_INPUT_WIDTH,
                          detections[i].y * MODEL_INPUT_HEIGHT);
            Serial.printf("  Size: %.1f x %.1f\n",
                          detections[i].w * MODEL_INPUT_WIDTH,
                          detections[i].h * MODEL_INPUT_HEIGHT);
        }
    }
    
    Serial.println("=======================================\n");
}

// ==================== SETUP ====================
void setup() {
    WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
    
    Serial.begin(115200);
    delay(1000);
    
    Serial.println("\n========================================");
    Serial.println("EE 4065 - YOLO Digit Detection");
    Serial.println("Student: KAAN ATALAY (150720057)");
    Serial.println("========================================\n");
    
    initCamera();
    initTFLite();
}

// ==================== MAIN LOOP ====================
void loop() {
    Serial.println("--- Starting Detection Cycle ---");
    
    // Capture image
    static uint8_t image_buffer[MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT * 2];
    if (!captureImage(image_buffer)) {
        delay(1000);
        return;
    }
    Serial.println("Image captured!");
    
    // Preprocess
    float* input_data = input->data.f;
    preprocessImage(image_buffer, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT, input_data);
    Serial.println("Image preprocessed!");
    
    // Run inference
    runInference();
    
    // Postprocess
    postprocessOutput();
    nonMaxSuppression();
    
    // Print results
    printResults();
    
    delay(2000);
}
