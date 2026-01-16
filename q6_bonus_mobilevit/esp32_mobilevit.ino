/*
 * EE 4065 - Embedded Digital Image Processing
 * Question 6 (BONUS): MobileViT on ESP32-CAM
 * 
 * Student: KAAN ATALAY
 * ID: 150720057
 * 
 * This code runs MobileViT inference on ESP32-CAM
 * for handwritten digit recognition.
 */

#include "esp_camera.h"
#include "Arduino.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

// TensorFlow Lite Micro
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Include the trained MobileViT model
#include "mobilevit_model.h"

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
#define NUM_CLASSES       10
#define TENSOR_ARENA_SIZE (350 * 1024)  // MobileViT needs more memory

// ==================== GLOBAL VARIABLES ====================
uint8_t* tensor_arena = nullptr;
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

uint8_t image_buffer[IMG_SIZE * IMG_SIZE];      // Grayscale
float input_data[IMG_SIZE * IMG_SIZE * 3];      // RGB (gray copied to 3 channels)

const char* digit_names[] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};

// ==================== FUNCTION PROTOTYPES ====================
void initCamera();
bool initTFLite();
bool captureImage();
void preprocessImage();
void runInference();
void printResults(float* predictions);

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

// ==================== TFLITE INITIALIZATION ====================
bool initTFLite() {
    // Allocate tensor arena in PSRAM
    tensor_arena = (uint8_t*)ps_malloc(TENSOR_ARENA_SIZE);
    if (!tensor_arena) {
        Serial.println("Failed to allocate tensor arena!");
        return false;
    }
    Serial.printf("Tensor arena allocated: %d bytes\n", TENSOR_ARENA_SIZE);
    
    // Load model
    model = tflite::GetModel(mobilevit_model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.printf("Model version mismatch: %d vs %d\n",
                      model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }
    
    // Create resolver with all ops (MobileViT uses many ops)
    static tflite::AllOpsResolver resolver;
    
    // Build interpreter
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, TENSOR_ARENA_SIZE, error_reporter);
    interpreter = &static_interpreter;
    
    // Allocate tensors
    TfLiteStatus status = interpreter->AllocateTensors();
    if (status != kTfLiteOk) {
        Serial.println("AllocateTensors() failed!");
        return false;
    }
    
    // Get input/output tensors
    input = interpreter->input(0);
    output = interpreter->output(0);
    
    Serial.println("TensorFlow Lite initialized!");
    Serial.printf("Model input: [%d, %d, %d, %d]\n",
                  input->dims->data[0], input->dims->data[1],
                  input->dims->data[2], input->dims->data[3]);
    Serial.printf("Arena used: %d bytes\n", interpreter->arena_used_bytes());
    
    return true;
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
     * Preprocess grayscale image for MobileViT:
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

// ==================== INFERENCE ====================
void runInference() {
    // Copy preprocessed data to input tensor
    float* input_tensor = input->data.f;
    memcpy(input_tensor, input_data, IMG_SIZE * IMG_SIZE * 3 * sizeof(float));
    
    // Run inference
    unsigned long start_time = millis();
    TfLiteStatus status = interpreter->Invoke();
    unsigned long inference_time = millis() - start_time;
    
    if (status != kTfLiteOk) {
        Serial.println("Inference failed!");
        return;
    }
    
    Serial.printf("Inference time: %lu ms\n", inference_time);
    
    // Get output
    float* predictions = output->data.f;
    printResults(predictions);
}

// ==================== PRINT RESULTS ====================
void printResults(float* predictions) {
    Serial.println("\n========== MOBILEVIT RESULTS ==========");
    
    // Find top prediction
    int top_class = 0;
    float top_confidence = predictions[0];
    
    for (int i = 1; i < NUM_CLASSES; i++) {
        if (predictions[i] > top_confidence) {
            top_confidence = predictions[i];
            top_class = i;
        }
    }
    
    // Print all predictions
    Serial.println("Class probabilities:");
    for (int i = 0; i < NUM_CLASSES; i++) {
        char bar[21];
        int bar_len = (int)(predictions[i] * 20);
        for (int j = 0; j < 20; j++) {
            bar[j] = (j < bar_len) ? '#' : '-';
        }
        bar[20] = '\0';
        
        Serial.printf("  %s: %.2f%% %s%s\n", 
                      digit_names[i], 
                      predictions[i] * 100,
                      bar,
                      (i == top_class) ? " <-- TOP" : "");
    }
    
    Serial.println("\n========================================");
    Serial.printf("RECOGNIZED DIGIT: %s (%.1f%% confidence)\n",
                  digit_names[top_class], top_confidence * 100);
    Serial.println("========================================\n");
    
    // Confidence assessment
    if (top_confidence > 0.9) {
        Serial.println("High confidence prediction!");
    } else if (top_confidence > 0.7) {
        Serial.println("Moderate confidence prediction.");
    } else if (top_confidence > 0.5) {
        Serial.println("Low confidence - consider retaking image.");
    } else {
        Serial.println("Very low confidence - result may be unreliable.");
    }
}

// ==================== SETUP ====================
void setup() {
    // Disable brownout detector
    WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
    
    Serial.begin(115200);
    delay(1000);
    
    Serial.println("\n========================================");
    Serial.println("EE 4065 - MobileViT Digit Recognition");
    Serial.println("Student: KAAN ATALAY (150720057)");
    Serial.println("========================================\n");
    
    Serial.println("Initializing camera...");
    initCamera();
    
    Serial.println("Initializing TensorFlow Lite...");
    if (!initTFLite()) {
        Serial.println("TFLite init failed! System halted.");
        while (1) delay(1000);
    }
    
    Serial.println("\nReady for digit recognition!");
}

// ==================== MAIN LOOP ====================
void loop() {
    Serial.println("\n=== MobileViT Recognition Cycle ===");
    
    // Capture image
    Serial.println("Capturing image...");
    if (!captureImage()) {
        delay(1000);
        return;
    }
    Serial.println("Image captured!");
    
    // Preprocess
    Serial.println("Preprocessing...");
    preprocessImage();
    
    // Run inference
    Serial.println("Running MobileViT inference...");
    runInference();
    
    // Wait before next cycle
    delay(3000);
}
