#include <Arduino.h>
#include <algorithm>
#include <math.h>
#include "../model_data.h"

static const uint32_t STREAM_MAGIC = 0x32535745UL;
static const uint32_t SAMPLE_MAGIC = 0x32535753UL;
static const int N_CHANNELS_EEG = 19;
static const int N_SAMPLES_EEG = 750;
static const int HOP_SAMPLES_EEG = 375;
static const int N_BASELINE = 15;
static const int N_WELCH = 500;
static const int N_FREQ = 251;

static float eeg[N_CHANNELS_EEG][N_SAMPLES_EEG];
static float ref_features[N_FEATURES];
static float features[N_FEATURES];
static float scaled_features[N_FEATURES];
static float baseline_delta[N_BASELINE];
static double window_sfreq_hz = 250.0;
static float hann_window[N_WELCH];
static float goertzel_coeff[N_FREQ];
static float median_tmp[N_SAMPLES_EEG];
static float reorder_tmp[N_CHANNELS_EEG][HOP_SAMPLES_EEG];
static uint32_t current_stream_id = 0;
static uint16_t current_model_index = 0;
static uint16_t current_stream_window_count = 0;
static uint32_t stream_sample_count = 0;
static uint32_t stream_window_index = 0;
static uint32_t expected_sequence = 0;
static bool have_sequence = false;
static uint32_t malformed_packets = 0;
static uint32_t dropped_sequences = 0;
static uint32_t buffer_overrun_count = 0;
static bool stream_configured = false;

enum RoiIndex { ROI_CENTRAL = 0, ROI_FRONTAL = 1, ROI_OCCIPITAL = 2, ROI_PARIETAL = 3, ROI_TEMPORAL = 4, N_ROIS = 5 };
enum BandIndex { BAND_DELTA = 0, BAND_THETA = 1, BAND_ALPHA = 2, BAND_BETA = 3, BAND_HIGHBETA = 4, N_BANDS = 5 };

static const int ROI_COUNTS[N_ROIS] = {3, 7, 2, 3, 4};
static const int ROI_IDXS[N_ROIS][7] = {
  {14, 13, 2, -1, -1, -1, -1},
  {18, 17, 16, 15, 8, 7, 1},
  {10, 9, -1, -1, -1, -1, -1},
  {12, 11, 0, -1, -1, -1, -1},
  {6, 5, 4, 3, -1, -1, -1},
};

struct RoiFeatures {
  float abs_power[N_BANDS];
  float rel_power[N_BANDS];
  float ratio_theta_alpha;
  float ratio_theta_beta;
  float ratio_alpha_beta;
  float var;
  float rms;
  float mav;
  float mean_power;
  float median_power;
  float line_length;
  float zc;
  float zc_rate;
  float ssc;
  float ssc_rate;
  float hjorth_activity;
  float hjorth_mobility;
  float hjorth_complexity;
  float spectral_entropy;
};

static RoiFeatures roi_features[N_ROIS];
static float ch_band[N_CHANNELS_EEG][N_BANDS];
static float ch_total[N_CHANNELS_EEG];
static float ch_entropy[N_CHANNELS_EEG];

static float round9(float value) {
  if (!isfinite(value)) {
    return NAN;
  }
  return roundf(value * 1000000000.0f) / 1000000000.0f;
}

static void init_welch_constants() {
  for (int n = 0; n < N_WELCH; n++) {
    hann_window[n] = 0.5f - 0.5f * cosf(2.0f * PI * float(n) / float(N_WELCH));
  }
  for (int k = 0; k < N_FREQ; k++) {
    goertzel_coeff[k] = 2.0f * cosf(2.0f * PI * float(k) / float(N_WELCH));
  }
}

static float median_power_channel(const float* x) {
  for (int i = 0; i < N_SAMPLES_EEG; i++) {
    median_tmp[i] = x[i] * x[i];
  }
  std::sort(median_tmp, median_tmp + N_SAMPLES_EEG);
  return 0.5f * (median_tmp[(N_SAMPLES_EEG / 2) - 1] + median_tmp[N_SAMPLES_EEG / 2]);
}

static int zero_crossings_channel(const float* x) {
  int count = 0;
  int prev_sign = 0;
  for (int i = 0; i < N_SAMPLES_EEG; i++) {
    int sign = (x[i] > 0.0f) ? 1 : ((x[i] < 0.0f) ? -1 : 0);
    if (sign == 0) {
      continue;
    }
    if (prev_sign != 0 && sign * prev_sign < 0) {
      count++;
    }
    prev_sign = sign;
  }
  return count;
}

static int slope_sign_changes_channel(const float* x) {
  int count = 0;
  for (int i = 1; i < N_SAMPLES_EEG - 1; i++) {
    const float left = x[i] - x[i - 1];
    const float right = x[i] - x[i + 1];
    if ((left * right) > 0.0f) {
      count++;
    }
  }
  return count;
}

static void hjorth_channel(const float* x, float& activity, float& mobility, float& complexity) {
  float mean_x = 0.0f;
  for (int i = 0; i < N_SAMPLES_EEG; i++) {
    mean_x += x[i];
  }
  mean_x /= float(N_SAMPLES_EEG);
  float var_x = 0.0f;
  for (int i = 0; i < N_SAMPLES_EEG; i++) {
    const float d = x[i] - mean_x;
    var_x += d * d;
  }
  var_x /= float(N_SAMPLES_EEG);

  const int n_dx = N_SAMPLES_EEG - 1;
  float mean_dx = 0.0f;
  for (int i = 0; i < n_dx; i++) {
    mean_dx += x[i + 1] - x[i];
  }
  mean_dx /= float(n_dx);
  float var_dx = 0.0f;
  for (int i = 0; i < n_dx; i++) {
    const float d = (x[i + 1] - x[i]) - mean_dx;
    var_dx += d * d;
  }
  var_dx /= float(n_dx);

  const int n_ddx = N_SAMPLES_EEG - 2;
  float mean_ddx = 0.0f;
  for (int i = 0; i < n_ddx; i++) {
    mean_ddx += x[i + 2] - 2.0f * x[i + 1] + x[i];
  }
  mean_ddx /= float(n_ddx);
  float var_ddx = 0.0f;
  for (int i = 0; i < n_ddx; i++) {
    const float d = (x[i + 2] - 2.0f * x[i + 1] + x[i]) - mean_ddx;
    var_ddx += d * d;
  }
  var_ddx /= float(n_ddx);

  activity = var_x;
  if (var_x <= 0.0f || var_dx <= 0.0f) {
    mobility = NAN;
    complexity = NAN;
    return;
  }
  mobility = sqrtf(var_dx / var_x);
  const float mobility_dx = sqrtf(var_ddx / var_dx);
  complexity = mobility > 0.0f ? mobility_dx / mobility : NAN;
}

static float goertzel_power(const float* x, int start, int k, float segment_mean) {
  const float coeff = goertzel_coeff[k];
  float s0 = 0.0f;
  float s1 = 0.0f;
  float s2 = 0.0f;
  for (int n = 0; n < N_WELCH; n++) {
    const float sample = (x[start + n] - segment_mean) * hann_window[n];
    s0 = sample + coeff * s1 - s2;
    s2 = s1;
    s1 = s0;
  }
  return s2 * s2 + s1 * s1 - coeff * s1 * s2;
}

static void compute_channel_psd_features(const float* x, int ch) {
  float psd_bins[81];
  for (int i = 0; i < 81; i++) {
    psd_bins[i] = 0.0f;
  }
  const int starts[2] = {0, 250};
  const double fs = window_sfreq_hz;
  const float win_power = 187.5f;
  for (int seg_i = 0; seg_i < 2; seg_i++) {
    const int start = starts[seg_i];
    float mean_seg = 0.0f;
    for (int n = 0; n < N_WELCH; n++) {
      mean_seg += x[start + n];
    }
    mean_seg /= float(N_WELCH);
    for (int k = 0; k <= 80; k++) {
      float power = goertzel_power(x, start, k, mean_seg);
      float psd = float(double(power) / (fs * double(win_power)));
      if (k > 0 && k < 250) {
        psd *= 2.0f;
      }
      psd_bins[k] += psd * 0.5f;
    }
  }

  auto integrate = [&](double fmin, double fmax) -> float {
    double total = 0.0;
    bool first = true;
    double prev = 0.0;
    double prev_freq = 0.0;
    for (int k = 0; k <= 80; k++) {
      const double freq = (double(k) * fs) / double(N_WELCH);
      if (!(freq >= fmin && freq < fmax)) {
        continue;
      }
      const double val = psd_bins[k];
      if (first) {
        prev = val;
        prev_freq = freq;
        first = false;
      } else {
        total += 0.5 * (prev + val) * (freq - prev_freq);
        prev = val;
        prev_freq = freq;
      }
    }
    return float(total);
  };

  ch_band[ch][BAND_DELTA] = integrate(1.0, 4.0);
  ch_band[ch][BAND_THETA] = integrate(4.0, 8.0);
  ch_band[ch][BAND_ALPHA] = integrate(8.0, 13.0);
  ch_band[ch][BAND_BETA] = integrate(13.0, 30.0);
  ch_band[ch][BAND_HIGHBETA] = integrate(30.0, 40.0);
  ch_total[ch] = integrate(1.0, 40.0);

  float sum_psd = 0.0f;
  int count = 0;
  for (int k = 0; k <= 80; k++) {
    const double freq = (double(k) * fs) / double(N_WELCH);
    if (!(freq >= 1.0 && freq < 40.0)) {
      continue;
    }
    const float v = psd_bins[k];
    if (isfinite(v) && v > 0.0f) {
      sum_psd += v;
      count++;
    }
  }
  if (sum_psd <= 0.0f || count < 2) {
    ch_entropy[ch] = NAN;
  } else {
    float ent = 0.0f;
    for (int k = 0; k <= 80; k++) {
      const double freq = (double(k) * fs) / double(N_WELCH);
      if (!(freq >= 1.0 && freq < 40.0)) {
        continue;
      }
      const float v = psd_bins[k];
      if (isfinite(v) && v > 0.0f) {
        const float p = v / sum_psd;
        ent -= p * logf(p);
      }
    }
    ch_entropy[ch] = ent / logf(float(count));
  }
}

static void compute_roi_features() {
  for (int ch = 0; ch < N_CHANNELS_EEG; ch++) {
    compute_channel_psd_features(eeg[ch], ch);
  }

  for (int roi = 0; roi < N_ROIS; roi++) {
    RoiFeatures& rf = roi_features[roi];
    memset(&rf, 0, sizeof(RoiFeatures));
    float roi_total = 0.0f;
    for (int b = 0; b < N_BANDS; b++) {
      float sum = 0.0f;
      for (int j = 0; j < ROI_COUNTS[roi]; j++) {
        sum += ch_band[ROI_IDXS[roi][j]][b];
      }
      rf.abs_power[b] = sum / float(ROI_COUNTS[roi]);
    }
    for (int j = 0; j < ROI_COUNTS[roi]; j++) {
      roi_total += ch_total[ROI_IDXS[roi][j]];
    }
    roi_total /= float(ROI_COUNTS[roi]);
    for (int b = 0; b < N_BANDS; b++) {
      rf.rel_power[b] = roi_total > 0.0f ? rf.abs_power[b] / roi_total : NAN;
    }
    rf.ratio_theta_alpha = rf.rel_power[BAND_ALPHA] > 0.0f ? rf.rel_power[BAND_THETA] / rf.rel_power[BAND_ALPHA] : NAN;
    rf.ratio_theta_beta = rf.rel_power[BAND_BETA] > 0.0f ? rf.rel_power[BAND_THETA] / rf.rel_power[BAND_BETA] : NAN;
    rf.ratio_alpha_beta = rf.rel_power[BAND_BETA] > 0.0f ? rf.rel_power[BAND_ALPHA] / rf.rel_power[BAND_BETA] : NAN;

    float sum_var = 0.0f, sum_rms = 0.0f, sum_mav = 0.0f, sum_mean_power = 0.0f, sum_median_power = 0.0f;
    float sum_ll = 0.0f, sum_zc = 0.0f, sum_zcr = 0.0f, sum_ssc = 0.0f, sum_sscr = 0.0f;
    float sum_hja = 0.0f, sum_hjm = 0.0f, sum_hjc = 0.0f, sum_ent = 0.0f;
    int count_hjm = 0, count_hjc = 0, count_ent = 0;
    for (int j = 0; j < ROI_COUNTS[roi]; j++) {
      const int ch = ROI_IDXS[roi][j];
      const float* x = eeg[ch];
      float mean_x = 0.0f;
      for (int i = 0; i < N_SAMPLES_EEG; i++) {
        mean_x += x[i];
      }
      mean_x /= float(N_SAMPLES_EEG);
      float var = 0.0f, abs_sum = 0.0f, sq_sum = 0.0f, ll = 0.0f;
      for (int i = 0; i < N_SAMPLES_EEG; i++) {
        const float centered = x[i] - mean_x;
        var += centered * centered;
        abs_sum += fabsf(x[i]);
        sq_sum += x[i] * x[i];
        if (i > 0) {
          ll += fabsf(x[i] - x[i - 1]);
        }
      }
      var /= float(N_SAMPLES_EEG);
      const float mean_power = sq_sum / float(N_SAMPLES_EEG);
      sum_var += var;
      sum_rms += sqrtf(mean_power);
      sum_mav += abs_sum / float(N_SAMPLES_EEG);
      sum_mean_power += mean_power;
      sum_median_power += median_power_channel(x);
      sum_ll += ll / float(N_SAMPLES_EEG - 1);
      const int zc = zero_crossings_channel(x);
      const int ssc = slope_sign_changes_channel(x);
      sum_zc += float(zc);
      sum_zcr += float(zc) / float(N_SAMPLES_EEG - 1);
      sum_ssc += float(ssc);
      sum_sscr += float(ssc) / float(N_SAMPLES_EEG - 2);
      float a, m, c;
      hjorth_channel(x, a, m, c);
      sum_hja += a;
      if (isfinite(m)) {
        sum_hjm += m;
        count_hjm++;
      }
      if (isfinite(c)) {
        sum_hjc += c;
        count_hjc++;
      }
      if (isfinite(ch_entropy[ch])) {
        sum_ent += ch_entropy[ch];
        count_ent++;
      }
    }
    const float denom = float(ROI_COUNTS[roi]);
    rf.var = sum_var / denom;
    rf.rms = sum_rms / denom;
    rf.mav = sum_mav / denom;
    rf.mean_power = sum_mean_power / denom;
    rf.median_power = sum_median_power / denom;
    rf.line_length = sum_ll / denom;
    rf.zc = sum_zc / denom;
    rf.zc_rate = sum_zcr / denom;
    rf.ssc = sum_ssc / denom;
    rf.ssc_rate = sum_sscr / denom;
    rf.hjorth_activity = sum_hja / denom;
    rf.hjorth_mobility = count_hjm > 0 ? sum_hjm / float(count_hjm) : NAN;
    rf.hjorth_complexity = count_hjc > 0 ? sum_hjc / float(count_hjc) : NAN;
    rf.spectral_entropy = count_ent > 0 ? sum_ent / float(count_ent) : NAN;
  }
}

static float band_delta(int roi, int band) {
  int band_offset = 0;
  if (band == BAND_THETA) band_offset = 0;
  else if (band == BAND_ALPHA) band_offset = 1;
  else if (band == BAND_BETA) band_offset = 2;
  return roi_features[roi].abs_power[band] - baseline_delta[roi * 3 + band_offset];
}

static void fill_feature_vector() {
  compute_roi_features();
  int i = 0;
  features[i++] = roi_features[ROI_CENTRAL].abs_power[BAND_ALPHA];
  features[i++] = band_delta(ROI_CENTRAL, BAND_ALPHA);
  features[i++] = roi_features[ROI_FRONTAL].abs_power[BAND_ALPHA];
  features[i++] = band_delta(ROI_FRONTAL, BAND_ALPHA);
  features[i++] = roi_features[ROI_OCCIPITAL].abs_power[BAND_ALPHA];
  features[i++] = band_delta(ROI_OCCIPITAL, BAND_ALPHA);
  features[i++] = roi_features[ROI_PARIETAL].abs_power[BAND_ALPHA];
  features[i++] = band_delta(ROI_PARIETAL, BAND_ALPHA);
  features[i++] = roi_features[ROI_TEMPORAL].abs_power[BAND_ALPHA];
  features[i++] = band_delta(ROI_TEMPORAL, BAND_ALPHA);
  for (int roi = 0; roi < N_ROIS; roi++) { features[i++] = roi_features[roi].abs_power[BAND_BETA]; features[i++] = band_delta(roi, BAND_BETA); }
  for (int roi = 0; roi < N_ROIS; roi++) features[i++] = roi_features[roi].abs_power[BAND_DELTA];
  for (int roi = 0; roi < N_ROIS; roi++) features[i++] = roi_features[roi].abs_power[BAND_HIGHBETA];
  for (int roi = 0; roi < N_ROIS; roi++) { features[i++] = roi_features[roi].abs_power[BAND_THETA]; features[i++] = band_delta(roi, BAND_THETA); }
  const float f3_alpha = ch_band[16][BAND_ALPHA];
  const float f4_alpha = ch_band[15][BAND_ALPHA];
  features[i++] = (f3_alpha > 0.0f && f4_alpha > 0.0f) ? logf(f4_alpha) - logf(f3_alpha) : NAN;
  features[i++] = ch_band[1][BAND_THETA];
  features[i++] = ch_total[1] > 0.0f ? ch_band[1][BAND_THETA] / ch_total[1] : NAN;
  for (int roi = 0; roi < N_ROIS; roi++) features[i++] = roi_features[roi].hjorth_activity;
  for (int roi = 0; roi < N_ROIS; roi++) features[i++] = roi_features[roi].hjorth_complexity;
  for (int roi = 0; roi < N_ROIS; roi++) features[i++] = roi_features[roi].hjorth_mobility;
  for (int roi = 0; roi < N_ROIS; roi++) features[i++] = roi_features[roi].line_length;
  for (int roi = 0; roi < N_ROIS; roi++) features[i++] = roi_features[roi].mav;
  for (int roi = 0; roi < N_ROIS; roi++) features[i++] = roi_features[roi].mean_power;
  for (int roi = 0; roi < N_ROIS; roi++) features[i++] = roi_features[roi].median_power;
  for (int roi = 0; roi < N_ROIS; roi++) features[i++] = roi_features[roi].ratio_alpha_beta;
  for (int roi = 0; roi < N_ROIS; roi++) features[i++] = roi_features[roi].ratio_theta_alpha;
  for (int roi = 0; roi < N_ROIS; roi++) features[i++] = roi_features[roi].ratio_theta_beta;
  for (int roi = 0; roi < N_ROIS; roi++) features[i++] = roi_features[roi].rel_power[BAND_ALPHA];
  for (int roi = 0; roi < N_ROIS; roi++) features[i++] = roi_features[roi].rel_power[BAND_BETA];
  for (int roi = 0; roi < N_ROIS; roi++) features[i++] = roi_features[roi].rel_power[BAND_DELTA];
  for (int roi = 0; roi < N_ROIS; roi++) features[i++] = roi_features[roi].rel_power[BAND_HIGHBETA];
  for (int roi = 0; roi < N_ROIS; roi++) features[i++] = roi_features[roi].rel_power[BAND_THETA];
  for (int roi = 0; roi < N_ROIS; roi++) features[i++] = roi_features[roi].rms;
  for (int roi = 0; roi < N_ROIS; roi++) features[i++] = roi_features[roi].spectral_entropy;
  features[i++] = roi_features[ROI_CENTRAL].ssc;
  features[i++] = roi_features[ROI_FRONTAL].ssc;
  features[i++] = roi_features[ROI_OCCIPITAL].ssc;
  features[i++] = roi_features[ROI_PARIETAL].ssc;
  for (int roi = 0; roi < N_ROIS; roi++) features[i++] = roi_features[roi].ssc_rate;
  features[i++] = roi_features[ROI_TEMPORAL].ssc;
  for (int roi = 0; roi < N_ROIS; roi++) features[i++] = roi_features[roi].var;
  features[i++] = roi_features[ROI_CENTRAL].zc;
  features[i++] = roi_features[ROI_FRONTAL].zc;
  features[i++] = roi_features[ROI_OCCIPITAL].zc;
  features[i++] = roi_features[ROI_PARIETAL].zc;
  for (int roi = 0; roi < N_ROIS; roi++) features[i++] = roi_features[roi].zc_rate;
  features[i++] = roi_features[ROI_TEMPORAL].zc;
  for (int j = 0; j < N_FEATURES; j++) {
    features[j] = round9(features[j]);
  }
}

static int classify(int model_index, const float* raw, float* logits, float* probs) {
  for (int i = 0; i < N_FEATURES; i++) {
    float value = raw[i];
    if (!isfinite(value)) value = IMPUTER_MEDIAN[model_index][i];
    const float lo = CLIP_LOWER[model_index][i];
    const float hi = CLIP_UPPER[model_index][i];
    if (isfinite(lo) && value < lo) value = lo;
    if (isfinite(hi) && value > hi) value = hi;
    float scale = SCALER_SCALE[model_index][i];
    if (!isfinite(scale) || scale == 0.0f) scale = 1.0f;
    scaled_features[i] = (value - SCALER_CENTER[model_index][i]) / scale;
  }
  for (int c = 0; c < N_CLASSES; c++) {
    float total = INTERCEPT[model_index][c];
    for (int j = 0; j < ACTIVE_COUNTS[model_index]; j++) {
      const int feature_index = ACTIVE_INDICES[model_index][j];
      total += COEF[model_index][c][j] * scaled_features[feature_index];
    }
    logits[c] = total;
  }
  float max_logit = logits[0];
  for (int c = 1; c < N_CLASSES; c++) if (logits[c] > max_logit) max_logit = logits[c];
  float denom = 0.0f;
  for (int c = 0; c < N_CLASSES; c++) {
    probs[c] = expf(logits[c] - max_logit);
    denom += probs[c];
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

static bool read_exact(uint8_t* dst, size_t n) {
  size_t got = 0;
  const uint32_t start = millis();
  while (got < n) {
    if (millis() - start > 10000UL) return false;
    int available = Serial.available();
    if (available <= 0) {
      delay(1);
      continue;
    }
    got += Serial.readBytes(dst + got, n - got);
  }
  return true;
}

static bool read_magic_resync(uint32_t& magic) {
  uint8_t buf[4] = {0, 0, 0, 0};
  if (!read_exact(buf, sizeof(buf))) return false;
  uint32_t skipped = 0;
  while (true) {
    magic = uint32_t(buf[0]) | (uint32_t(buf[1]) << 8) | (uint32_t(buf[2]) << 16) | (uint32_t(buf[3]) << 24);
    if (magic == STREAM_MAGIC || magic == SAMPLE_MAGIC) {
      if (skipped > 0) {
        malformed_packets++;
      }
      return true;
    }
    buf[0] = buf[1];
    buf[1] = buf[2];
    buf[2] = buf[3];
    if (!read_exact(&buf[3], 1)) return false;
    skipped++;
  }
}

void setup() {
  Serial.setRxBufferSize(32768);
  Serial.begin(921600);
  Serial.setTimeout(10000);
  delay(500);
  init_welch_constants();
  Serial.println("# esp32_continuous_replay ready");
  Serial.println("# demo_type=real-time embedded inference on replayed/pre-recorded preprocessed EEG samples");
  Serial.println("# columns=stream_id,window_index,start_sample_index,end_sample_index,model_index,pred_label,logit_0,logit_1,logit_2,prob_0,prob_1,prob_2,feature_us,classifier_us,total_us,malformed_packets,dropped_sequences,buffer_overrun_count");
}

static void reset_stream_state(uint32_t stream_id, uint16_t model_index, uint16_t window_count, double sfreq_hz) {
  if (stream_id == 0) {
    malformed_packets = 0;
    dropped_sequences = 0;
    buffer_overrun_count = 0;
    have_sequence = false;
    expected_sequence = 0;
  }
  current_stream_id = stream_id;
  current_model_index = model_index;
  current_stream_window_count = window_count;
  window_sfreq_hz = (isfinite(sfreq_hz) && sfreq_hz > 0.0) ? sfreq_hz : 250.0;
  stream_sample_count = 0;
  stream_window_index = 0;
  stream_configured = true;
  memset(eeg, 0, sizeof(eeg));
}

static bool prepare_window_for_feature_extraction(uint32_t end_sample_index) {
  const uint32_t start_sample_index = end_sample_index + 1UL - uint32_t(N_SAMPLES_EEG);
  const int start_pos = int(start_sample_index % uint32_t(N_SAMPLES_EEG));
  if (start_pos == 0) {
    return false;
  }
  if (start_pos != HOP_SAMPLES_EEG) {
    buffer_overrun_count++;
    return false;
  }
  for (int ch = 0; ch < N_CHANNELS_EEG; ch++) {
    memcpy(reorder_tmp[ch], &eeg[ch][0], sizeof(float) * HOP_SAMPLES_EEG);
    memmove(&eeg[ch][0], &eeg[ch][HOP_SAMPLES_EEG], sizeof(float) * HOP_SAMPLES_EEG);
    memcpy(&eeg[ch][HOP_SAMPLES_EEG], reorder_tmp[ch], sizeof(float) * HOP_SAMPLES_EEG);
  }
  return true;
}

static void restore_ring_after_feature_extraction(bool rotated) {
  if (!rotated) return;
  for (int ch = 0; ch < N_CHANNELS_EEG; ch++) {
    memcpy(reorder_tmp[ch], &eeg[ch][0], sizeof(float) * HOP_SAMPLES_EEG);
    memmove(&eeg[ch][0], &eeg[ch][HOP_SAMPLES_EEG], sizeof(float) * HOP_SAMPLES_EEG);
    memcpy(&eeg[ch][HOP_SAMPLES_EEG], reorder_tmp[ch], sizeof(float) * HOP_SAMPLES_EEG);
  }
}

static void classify_current_window(uint32_t window_index, uint32_t end_sample_index) {
  const uint32_t start_sample_index = end_sample_index + 1UL - uint32_t(N_SAMPLES_EEG);
  const uint32_t total_start = micros();
  const uint32_t feature_start = micros();
  const bool rotated = prepare_window_for_feature_extraction(end_sample_index);
  fill_feature_vector();
  restore_ring_after_feature_extraction(rotated);
  const uint32_t feature_us = micros() - feature_start;

  float logits[N_CLASSES] = {0};
  float probs[N_CLASSES] = {0};
  const uint32_t classifier_start = micros();
  const int pred = classify(current_model_index, features, logits, probs);
  const uint32_t classifier_us = micros() - classifier_start;
  const uint32_t total_us = micros() - total_start;

  Serial.print(current_stream_id);
  Serial.print(',');
  Serial.print(window_index);
  Serial.print(',');
  Serial.print(start_sample_index);
  Serial.print(',');
  Serial.print(end_sample_index);
  Serial.print(',');
  Serial.print(current_model_index);
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
  Serial.print(feature_us);
  Serial.print(',');
  Serial.print(classifier_us);
  Serial.print(',');
  Serial.print(total_us);
  Serial.print(',');
  Serial.print(malformed_packets);
  Serial.print(',');
  Serial.print(dropped_sequences);
  Serial.print(',');
  Serial.print(buffer_overrun_count);
  Serial.println();
}

static void handle_stream_header() {
  uint32_t stream_id = 0;
  uint16_t model_index = 0;
  uint16_t window_count = 0;
  double sfreq_hz = 250.0;
  if (!read_exact(reinterpret_cast<uint8_t*>(&stream_id), sizeof(stream_id))) return;
  if (!read_exact(reinterpret_cast<uint8_t*>(&model_index), sizeof(model_index))) return;
  if (!read_exact(reinterpret_cast<uint8_t*>(&window_count), sizeof(window_count))) return;
  if (!read_exact(reinterpret_cast<uint8_t*>(&sfreq_hz), sizeof(sfreq_hz))) return;
  if (model_index >= N_MODELS) {
    malformed_packets++;
    stream_configured = false;
    Serial.println("# bad_model_index");
    return;
  }
  if (!read_exact(reinterpret_cast<uint8_t*>(baseline_delta), sizeof(baseline_delta))) return;
  reset_stream_state(stream_id, model_index, window_count, sfreq_hz);
}

static void handle_sample_packet() {
  uint32_t sequence = 0;
  uint32_t sample_index = 0;
  uint16_t stream_id = 0;
  uint16_t flags = 0;
  float sample[N_CHANNELS_EEG];
  if (!read_exact(reinterpret_cast<uint8_t*>(&sequence), sizeof(sequence))) return;
  if (!read_exact(reinterpret_cast<uint8_t*>(&sample_index), sizeof(sample_index))) return;
  if (!read_exact(reinterpret_cast<uint8_t*>(&stream_id), sizeof(stream_id))) return;
  if (!read_exact(reinterpret_cast<uint8_t*>(&flags), sizeof(flags))) return;
  if (!read_exact(reinterpret_cast<uint8_t*>(sample), sizeof(sample))) return;

  if (!stream_configured || uint32_t(stream_id) != current_stream_id) {
    malformed_packets++;
    return;
  }

  if (!have_sequence) {
    expected_sequence = sequence;
    have_sequence = true;
  }
  if (sequence != expected_sequence) {
    if (sequence > expected_sequence) {
      dropped_sequences += sequence - expected_sequence;
    } else {
      malformed_packets++;
    }
    expected_sequence = sequence;
  }
  expected_sequence++;

  if (sample_index != stream_sample_count) {
    if (sample_index > stream_sample_count) {
      dropped_sequences += sample_index - stream_sample_count;
    } else {
      malformed_packets++;
    }
    stream_sample_count = sample_index;
  }

  const int ring_pos = int(sample_index % uint32_t(N_SAMPLES_EEG));
  for (int ch = 0; ch < N_CHANNELS_EEG; ch++) {
    eeg[ch][ring_pos] = sample[ch];
  }
  stream_sample_count++;

  const uint32_t samples_ready = sample_index + 1UL;
  if (samples_ready >= uint32_t(N_SAMPLES_EEG) && ((samples_ready - uint32_t(N_SAMPLES_EEG)) % uint32_t(HOP_SAMPLES_EEG) == 0)) {
    if (current_stream_window_count == 0 || stream_window_index < uint32_t(current_stream_window_count)) {
      classify_current_window(stream_window_index, sample_index);
      stream_window_index++;
    } else {
      buffer_overrun_count++;
    }
  }

  (void)flags;
}

void loop() {
  uint32_t magic = 0;
  if (!read_magic_resync(magic)) return;
  if (magic == STREAM_MAGIC) {
    handle_stream_header();
  } else if (magic == SAMPLE_MAGIC) {
    handle_sample_packet();
  }
}
