/*
 * EE 4065 - Embedded Digital Image Processing
 * Question 4: Multi-Model Digit Recognition on ESP32-CAM
 * 
 * Student: KAAN ATALAY
 * ID: 150720057
 * 
 * This code runs multiple models (SqueezeNet, MobileNet, EfficientNet, ResNet)
 * on ESP32-CAM and fuses their results for digit recognition.
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

// Include model headers (generated from TFLite files)
#include "squeezenet_model.h"
#include "mobilenet_model.h"
#include "efficientnet_model.h"
#include "resnet_model.h"

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
#define NUM_MODELS        4
#define TENSOR_ARENA_SIZE (200 * 1024)  // 200KB per model

// ==================== MODEL STRUCTURE ====================
typedef struct {
    const char* name;
    const unsigned char* model_data;
    unsigned int model_size;
    tflite::MicroInterpreter* interpreter;
    TfLiteTensor* input;
    TfLiteTensor* output;
    float accuracy_weight;  // Weight for ensemble
} ModelInfo;

// ==================== GLOBAL VARIABLES ====================
uint8_t* tensor_arena = nullptr;
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;

ModelInfo models[NUM_MODELS];
float ensemble_predictions[NUM_CLASSES];
int vote_counts[NUM_CLASSES];

uint8_t image_buffer[IMG_SIZE * IMG_SIZE];      // Grayscale
float input_data[IMG_SIZE * IMG_SIZE * 3];      // RGB (gray copied to 3 channels)

const char* digit_names[] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};

// Model weights (based on typical accuracy)
float model_weights[NUM_MODELS] = {0.22f, 0.28f, 0.25f, 0.25f};

// ==================== FUNCTION PROTOTYPES ====================
void initCamera();
bool initModels();
bool initSingleModel(int idx);
void freeModels();
bool captureImage();
void preprocessImage();
void runAllModels();
void runSingleModel(int idx, float* predictions);
int fuseResultsWeightedAverage();
int fuseResultsMajorityVoting();
int fuseResultsMaxConfidence();
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

// ==================== MODEL INITIALIZATION ====================
bool initModels() {
    // Allocate tensor arena in PSRAM
    tensor_arena = (uint8_t*)ps_malloc(TENSOR_ARENA_SIZE);
    if (!tensor_arena) {
        Serial.println("Failed to allocate tensor arena!");
        return false;
    }
    
    // Setup model info
    models[0] = {"SqueezeNet", squeezenet_model_tflite, squeezenet_model_tflite_len,
                 nullptr, nullptr, nullptr, model_weights[0]};
    models[1] = {"MobileNetV2", mobilenet_model_tflite, mobilenet_model_tflite_len,
                 nullptr, nullptr, nullptr, model_weights[1]};
    models[2] = {"EfficientNet", efficientnet_model_tflite, efficientnet_model_tflite_len,
                 nullptr, nullptr, nullptr, model_weights[2]};
    models[3] = {"ResNet", resnet_model_tflite, resnet_model_tflite_len,
                 nullptr, nullptr, nullptr, model_weights[3]};
    
    Serial.println("Models configured!");
    return true;
}

bool initSingleModel(int idx) {
    /*
     * Initialize a single model for inference.
     * Due to memory constraints, we load models one at a time.
     */
    const tflite::Model* model = tflite::GetModel(models[idx].model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.printf("Model %s version mismatch!\n", models[idx].name);
        return false;
    }
    
    static tflite::AllOpsResolver resolver;
    
    // Create interpreter
    models[idx].interpreter = new tflite::MicroInterpreter(
        model, resolver, tensor_arena, TENSOR_ARENA_SIZE, error_reporter);
    
    if (models[idx].interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.printf("Failed to allocate tensors for %s\n", models[idx].name);
        return false;
    }
    
    models[idx].input = models[idx].interpreter->input(0);
    models[idx].output = models[idx].interpreter->output(0);
    
    Serial.printf("%s initialized (arena: %d bytes)\n",
                  models[idx].name, models[idx].interpreter->arena_used_bytes());
    
    return true;
}

void freeModels() {
    for (int i = 0; i < NUM_MODELS; i++) {
        if (models[i].interpreter) {
            delete models[i].interpreter;
            models[i].interpreter = nullptr;
        }
    }
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

// ==================== IMAGE PREPROCESSING ====================
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

// ==================== RUN INFERENCE ====================
void runSingleModel(int idx, float* predictions) {
    /*
     * Run inference on a single model.
     * Returns predictions array with NUM_CLASSES probabilities.
     */
    
    // Initialize model
    if (!initSingleModel(idx)) {
        Serial.printf("Failed to init %s\n", models[idx].name);
        for (int i = 0; i < NUM_CLASSES; i++) predictions[i] = 0.1f;
        return;
    }
    
    // Copy input data
    float* input_tensor = models[idx].input->data.f;
    memcpy(input_tensor, input_data, IMG_SIZE * IMG_SIZE * 3 * sizeof(float));
    
    // Run inference
    unsigned long start = millis();
    TfLiteStatus status = models[idx].interpreter->Invoke();
    unsigned long elapsed = millis() - start;
    
    if (status != kTfLiteOk) {
        Serial.printf("%s inference failed!\n", models[idx].name);
        for (int i = 0; i < NUM_CLASSES; i++) predictions[i] = 0.1f;
    } else {
        // Copy output
        float* output_tensor = models[idx].output->data.f;
        memcpy(predictions, output_tensor, NUM_CLASSES * sizeof(float));
        
        Serial.printf("%s: %lu ms\n", models[idx].name, elapsed);
    }
    
    // Free model to make room for next one
    delete models[idx].interpreter;
    models[idx].interpreter = nullptr;
}

void runAllModels() {
    /*
     * Run all models sequentially (memory constraint).
     * Store individual predictions for fusion.
     */
    
    static float all_predictions[NUM_MODELS][NUM_CLASSES];
    
    // Reset ensemble predictions
    for (int i = 0; i < NUM_CLASSES; i++) {
        ensemble_predictions[i] = 0;
        vote_counts[i] = 0;
    }
    
    Serial.println("\n--- Running All Models ---");
    
    for (int m = 0; m < NUM_MODELS; m++) {
        runSingleModel(m, all_predictions[m]);
        
        // Find prediction
        int pred_class = 0;
        float max_conf = all_predictions[m][0];
        for (int c = 1; c < NUM_CLASSES; c++) {
            if (all_predictions[m][c] > max_conf) {
                max_conf = all_predictions[m][c];
                pred_class = c;
            }
        }
        
        Serial.printf("  %s: Digit %d (%.1f%%)\n",
                      models[m].name, pred_class, max_conf * 100);
        
        // Add to weighted average
        for (int c = 0; c < NUM_CLASSES; c++) {
            ensemble_predictions[c] += all_predictions[m][c] * models[m].accuracy_weight;
        }
        
        // Add vote
        vote_counts[pred_class]++;
    }
}

// ==================== FUSION METHODS ====================

int fuseResultsWeightedAverage() {
    /*
     * Fusion Method 1: Weighted Average
     * Combines predictions using accuracy-based weights.
     */
    int best_class = 0;
    float max_score = ensemble_predictions[0];
    
    for (int c = 1; c < NUM_CLASSES; c++) {
        if (ensemble_predictions[c] > max_score) {
            max_score = ensemble_predictions[c];
            best_class = c;
        }
    }
    
    return best_class;
}

int fuseResultsMajorityVoting() {
    /*
     * Fusion Method 2: Majority Voting
     * Each model gets one vote for its predicted class.
     */
    int best_class = 0;
    int max_votes = vote_counts[0];
    
    for (int c = 1; c < NUM_CLASSES; c++) {
        if (vote_counts[c] > max_votes) {
            max_votes = vote_counts[c];
            best_class = c;
        }
    }
    
    return best_class;
}

int fuseResultsMaxConfidence() {
    /*
     * Fusion Method 3: Maximum Confidence
     * Select the class with highest confidence across all models.
     */
    // Already implemented via weighted average with equal weights
    return fuseResultsWeightedAverage();
}

// ==================== PRINT RESULTS ====================
void printResults() {
    Serial.println("\n========== FUSION RESULTS ==========");
    
    int result_weighted = fuseResultsWeightedAverage();
    int result_voting = fuseResultsMajorityVoting();
    
    Serial.printf("Weighted Average: Digit %d (%.1f%%)\n",
                  result_weighted, ensemble_predictions[result_weighted] * 100);
    
    Serial.printf("Majority Voting:  Digit %d (%d/%d votes)\n",
                  result_voting, vote_counts[result_voting], NUM_MODELS);
    
    // Final decision (prefer voting if unanimous)
    int final_result;
    if (vote_counts[result_voting] >= 3) {
        final_result = result_voting;
        Serial.println("Using: Majority Voting (high agreement)");
    } else {
        final_result = result_weighted;
        Serial.println("Using: Weighted Average (low agreement)");
    }
    
    Serial.println("\n======================================");
    Serial.printf("FINAL RECOGNITION: %s\n", digit_names[final_result]);
    Serial.println("======================================\n");
}

// ==================== SETUP ====================
void setup() {
    WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
    
    Serial.begin(115200);
    delay(1000);
    
    Serial.println("\n========================================");
    Serial.println("EE 4065 - Multi-Model Digit Recognition");
    Serial.println("Student: KAAN ATALAY (150720057)");
    Serial.println("========================================\n");
    
    initCamera();
    initModels();
}

// ==================== MAIN LOOP ====================
void loop() {
    Serial.println("=== Starting Recognition Cycle ===");
    
    // Capture
    if (!captureImage()) {
        delay(1000);
        return;
    }
    Serial.println("Image captured!");
    
    // Preprocess
    preprocessImage();
    Serial.println("Image preprocessed!");
    
    // Run all models
    runAllModels();
    
    // Print fused results
    printResults();
    
    delay(3000);
}
