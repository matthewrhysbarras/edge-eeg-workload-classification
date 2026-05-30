#include <Arduino.h>
#include <math.h>
#include "model_data.h"

static const size_t LINE_BUFFER_SIZE = 8192;
static char line_buffer[LINE_BUFFER_SIZE];
static float raw_features[N_FEATURES];
static float scaled_features[N_FEATURES];

static bool parse_line(char* line, long& replay_row_index, int& model_index) {
  char* saveptr = nullptr;
  char* token = strtok_r(line, ",", &saveptr);
  if (token == nullptr) {
    return false;
  }
  replay_row_index = strtol(token, nullptr, 10);

  token = strtok_r(nullptr, ",", &saveptr);
  if (token == nullptr) {
    return false;
  }
  model_index = atoi(token);
  if (model_index < 0 || model_index >= N_MODELS) {
    return false;
  }

  for (int i = 0; i < N_FEATURES; i++) {
    token = strtok_r(nullptr, ",", &saveptr);
    if (token == nullptr) {
      return false;
    }
    raw_features[i] = strtof(token, nullptr);
  }
  return true;
}

static int predict(int model_index, const float* raw, float* logits, float* probs) {
  for (int i = 0; i < N_FEATURES; i++) {
    float value = raw[i];
    if (!isfinite(value)) {
      value = IMPUTER_MEDIAN[model_index][i];
    }
    const float lo = CLIP_LOWER[model_index][i];
    const float hi = CLIP_UPPER[model_index][i];
    if (isfinite(lo) && value < lo) {
      value = lo;
    }
    if (isfinite(hi) && value > hi) {
      value = hi;
    }
    float scale = SCALER_SCALE[model_index][i];
    if (!isfinite(scale) || scale == 0.0f) {
      scale = 1.0f;
    }
    scaled_features[i] = (value - SCALER_CENTER[model_index][i]) / scale;
  }

  for (int c = 0; c < N_CLASSES; c++) {
    float total = INTERCEPT[model_index][c];
    for (int j = 0; j < ACTIVE_COUNTS[model_index]; j++) {
      const int feature_index = ACTIVE_INDICES[model_index][j];
      if (feature_index >= 0) {
        total += COEF[model_index][c][j] * scaled_features[feature_index];
      }
    }
    logits[c] = total;
  }

  float max_logit = logits[0];
  for (int c = 1; c < N_CLASSES; c++) {
    if (logits[c] > max_logit) {
      max_logit = logits[c];
    }
  }

  float denom = 0.0f;
  for (int c = 0; c < N_CLASSES; c++) {
    probs[c] = expf(logits[c] - max_logit);
    denom += probs[c];
  }
  if (denom <= 0.0f || !isfinite(denom)) {
    denom = 1.0f;
  }

  int best = 0;
  float best_prob = -1.0f;
  for (int c = 0; c < N_CLASSES; c++) {
    probs[c] /= denom;
    if (probs[c] > best_prob) {
      best_prob = probs[c];
      best = c;
    }
  }
  return best;
}

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(5000);
  delay(500);
  Serial.println("# esp32_feature_replay ready");
  Serial.println("# demo_type=real-time embedded inference on replayed/pre-recorded feature vectors");
  Serial.println("# columns=replay_row_index,model_index,pred_label,logit_0,logit_1,logit_2,prob_0,prob_1,prob_2,inference_us");
}

void loop() {
  const size_t n = Serial.readBytesUntil('\n', line_buffer, LINE_BUFFER_SIZE - 1);
  if (n == 0) {
    return;
  }
  line_buffer[n] = '\0';

  long replay_row_index = -1;
  int model_index = -1;
  if (!parse_line(line_buffer, replay_row_index, model_index)) {
    Serial.println("# parse_error");
    return;
  }

  float logits[N_CLASSES] = {0};
  float probs[N_CLASSES] = {0};
  const uint32_t start_us = micros();
  const int pred = predict(model_index, raw_features, logits, probs);
  const uint32_t elapsed_us = micros() - start_us;

  Serial.print(replay_row_index);
  Serial.print(',');
  Serial.print(model_index);
  Serial.print(',');
  Serial.print(CLASS_LABELS[pred]);
  for (int c = 0; c < N_CLASSES; c++) {
    Serial.print(',');
    Serial.print(logits[c], 7);
  }
  for (int c = 0; c < N_CLASSES; c++) {
    Serial.print(',');
    Serial.print(probs[c], 7);
  }
  Serial.print(',');
  Serial.println(elapsed_us);
}
