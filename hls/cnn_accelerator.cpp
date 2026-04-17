// ============================================================================
// Hand-Gesture CNN \u2014 AXI4-Stream (Full: Blocks 1\u20134 + Dense1 + Dense2)
// ----------------------------------------------------------------------------
// - AXIS in/out with TKEEP + TLAST
// - AXI-Lite control: img_words (write-only), debug_flag (readable)
// - Resource-conscious: feature maps in BRAM, weights in ROM, no wide pipelining
// - Outputs 6 signed int8 logits (each packed into low 8 bits of a 32-bit word)
// ============================================================================

#include <ap_int.h>
#include <hls_stream.h>
#include <math.h>
#include "ap_axi_sdata.h"

// ===== Weights (all) =====
#include "depthwise_conv2d_w0.h"
#include "batch_normalization_w1.h"
#include "batch_normalization_w2.h"
#include "batch_normalization_w3.h"
#include "batch_normalization_w4.h"
#include "conv2d_w5.h"
#include "batch_normalization_1_w6.h"
#include "batch_normalization_1_w7.h"
#include "batch_normalization_1_w8.h"
#include "batch_normalization_1_w9.h"

#include "depthwise_conv2d_1_w10.h"
#include "batch_normalization_2_w11.h"
#include "batch_normalization_2_w12.h"
#include "batch_normalization_2_w13.h"
#include "batch_normalization_2_w14.h"
#include "conv2d_1_w15.h"
#include "batch_normalization_3_w16.h"
#include "batch_normalization_3_w17.h"
#include "batch_normalization_3_w18.h"
#include "batch_normalization_3_w19.h"

#include "depthwise_conv2d_2_w20.h"
#include "batch_normalization_4_w21.h"
#include "batch_normalization_4_w22.h"
#include "batch_normalization_4_w23.h"
#include "batch_normalization_4_w24.h"
#include "conv2d_2_w25.h"
#include "batch_normalization_5_w26.h"
#include "batch_normalization_5_w27.h"
#include "batch_normalization_5_w28.h"
#include "batch_normalization_5_w29.h"

#include "depthwise_conv2d_3_w30.h"
#include "batch_normalization_6_w31.h"
#include "batch_normalization_6_w32.h"
#include "batch_normalization_6_w33.h"
#include "batch_normalization_6_w34.h"
#include "conv2d_3_w35.h"
#include "batch_normalization_7_w36.h"
#include "batch_normalization_7_w37.h"
#include "batch_normalization_7_w38.h"
#include "batch_normalization_7_w39.h"

#include "dense_w40.h"
#include "dense_w41.h"
#include "dense_1_w42.h"
#include "dense_1_w43.h"

#define IMG_SIZE   64
#define N_CLASSES  6

typedef ap_axiu<32,1,1,1> axis_t;

// ========================== Utility =========================================
inline int8_t clamp_int8(int32_t x){
    if(x>127) return 127;
    if(x<-128) return -128;
    return (int8_t)x;
}
inline int8_t maxpool2x2_int8(int8_t w[2][2]){
    int8_t m=w[0][0];
    for(int i=0;i<2;i++)
        for(int j=0;j<2;j++)
            if(w[i][j]>m) m=w[i][j];
    return m;
}
static inline int8_t apply_bn_relu(const int32_t acc,
                                   const int8_t g,const int8_t b,
                                   const int8_t m,const int8_t v){
    const float eps=1e-5f;
    float gamma=((float)g)/127.0f;
    float beta =((float)b)/127.0f;
    float mean =((float)m)/127.0f;
    float var  =((float)v)/127.0f;
    float x=((float)acc)/128.0f;
    float y=gamma*((x-mean)/sqrtf(var+eps))+beta;
    int32_t q=(int32_t)roundf(y*127.0f);
    if(q<0) q=0;
    return clamp_int8(q);
}
static inline int8_t apply_bn(const int32_t acc,
                              const int8_t g,const int8_t b,
                              const int8_t m,const int8_t v){
    const float eps=1e-5f;
    float gamma=((float)g)/127.0f;
    float beta =((float)b)/127.0f;
    float mean =((float)m)/127.0f;
    float var  =((float)v)/127.0f;
    float x=((float)acc)/128.0f;
    float y=gamma*((x-mean)/sqrtf(var+eps))+beta;
    int32_t q=(int32_t)roundf(y*127.0f);
    return clamp_int8(q);
}

// ============================ Top Function ==================================
void cnn_accel_axis(
    hls::stream<axis_t>& s_axis,
    hls::stream<axis_t>& m_axis,
    unsigned int img_words,       // should be 64*64*3 = 12288
    unsigned int &debug_flag
){
#pragma HLS INTERFACE axis port=s_axis
#pragma HLS INTERFACE axis port=m_axis
#pragma HLS INTERFACE s_axilite port=img_words bundle=CTRL
#pragma HLS INTERFACE s_axilite port=debug_flag bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return    bundle=CTRL

    debug_flag = 0;

    // -------- Feature maps (BRAM) --------
    static uint8_t img[IMG_SIZE][IMG_SIZE][3];
    static int8_t  dw1_out[IMG_SIZE][IMG_SIZE][3];
    static int8_t  block1_pw[IMG_SIZE][IMG_SIZE][8];
    static int8_t  block1_pool[IMG_SIZE/2][IMG_SIZE/2][8];

    static int8_t  dw2_out[IMG_SIZE/2][IMG_SIZE/2][8];
    static int8_t  block2_pw[IMG_SIZE/2][IMG_SIZE/2][16];
    static int8_t  block2_pool[IMG_SIZE/4][IMG_SIZE/4][16];

    static int8_t  dw3_out[IMG_SIZE/4][IMG_SIZE/4][16];
    static int8_t  block3_pw[IMG_SIZE/4][IMG_SIZE/4][32];
    static int8_t  block3_pool[IMG_SIZE/8][IMG_SIZE/8][32];

    static int8_t  dw4_out[IMG_SIZE/8][IMG_SIZE/8][32];
    static int8_t  block4_pw[IMG_SIZE/8][IMG_SIZE/8][32];
    static int8_t  block4_pool[IMG_SIZE/16][IMG_SIZE/16][32];

    static int8_t  flat[512];
    static int8_t  fc1[32];
    static int8_t  output[N_CLASSES];

#pragma HLS BIND_STORAGE variable=img           type=ram_t2p impl=bram
#pragma HLS BIND_STORAGE variable=dw1_out       type=ram_t2p impl=bram
#pragma HLS BIND_STORAGE variable=block1_pw     type=ram_t2p impl=bram
#pragma HLS BIND_STORAGE variable=block1_pool   type=ram_t2p impl=bram

#pragma HLS BIND_STORAGE variable=dw2_out       type=ram_t2p impl=bram
#pragma HLS BIND_STORAGE variable=block2_pw     type=ram_t2p impl=bram
#pragma HLS BIND_STORAGE variable=block2_pool   type=ram_t2p impl=bram

#pragma HLS BIND_STORAGE variable=dw3_out       type=ram_t2p impl=bram
#pragma HLS BIND_STORAGE variable=block3_pw     type=ram_t2p impl=bram
#pragma HLS BIND_STORAGE variable=block3_pool   type=ram_t2p impl=bram

#pragma HLS BIND_STORAGE variable=dw4_out       type=ram_t2p impl=bram
#pragma HLS BIND_STORAGE variable=block4_pw     type=ram_t2p impl=bram
#pragma HLS BIND_STORAGE variable=block4_pool   type=ram_t2p impl=bram

#pragma HLS BIND_STORAGE variable=flat          type=ram_t2p impl=bram
#pragma HLS BIND_STORAGE variable=fc1           type=ram_t2p impl=bram
#pragma HLS BIND_STORAGE variable=output        type=ram_1p  impl=bram

    // -------- Bind weights to ROM --------
#pragma HLS RESOURCE variable=depthwise_conv2d_w0 core=ROM_1P
#pragma HLS RESOURCE variable=batch_normalization_w1 core=ROM_1P
#pragma HLS RESOURCE variable=batch_normalization_w2 core=ROM_1P
#pragma HLS RESOURCE variable=batch_normalization_w3 core=ROM_1P
#pragma HLS RESOURCE variable=batch_normalization_w4 core=ROM_1P
#pragma HLS RESOURCE variable=conv2d_w5 core=ROM_1P
#pragma HLS RESOURCE variable=batch_normalization_1_w6 core=ROM_1P
#pragma HLS RESOURCE variable=batch_normalization_1_w7 core=ROM_1P
#pragma HLS RESOURCE variable=batch_normalization_1_w8 core=ROM_1P
#pragma HLS RESOURCE variable=batch_normalization_1_w9 core=ROM_1P

#pragma HLS RESOURCE variable=depthwise_conv2d_1_w10 core=ROM_1P
#pragma HLS RESOURCE variable=batch_normalization_2_w11 core=ROM_1P
#pragma HLS RESOURCE variable=batch_normalization_2_w12 core=ROM_1P
#pragma HLS RESOURCE variable=batch_normalization_2_w13 core=ROM_1P
#pragma HLS RESOURCE variable=batch_normalization_2_w14 core=ROM_1P
#pragma HLS RESOURCE variable=conv2d_1_w15 core=ROM_1P
#pragma HLS RESOURCE variable=batch_normalization_3_w16 core=ROM_1P
#pragma HLS RESOURCE variable=batch_normalization_3_w17 core=ROM_1P
#pragma HLS RESOURCE variable=batch_normalization_3_w18 core=ROM_1P
#pragma HLS RESOURCE variable=batch_normalization_3_w19 core=ROM_1P

#pragma HLS RESOURCE variable=depthwise_conv2d_2_w20 core=ROM_1P
#pragma HLS RESOURCE variable=batch_normalization_4_w21 core=ROM_1P
#pragma HLS RESOURCE variable=batch_normalization_4_w22 core=ROM_1P
#pragma HLS RESOURCE variable=batch_normalization_4_w23 core=ROM_1P
#pragma HLS RESOURCE variable=batch_normalization_4_w24 core=ROM_1P
#pragma HLS RESOURCE variable=conv2d_2_w25 core=ROM_1P
#pragma HLS RESOURCE variable=batch_normalization_5_w26 core=ROM_1P
#pragma HLS RESOURCE variable=batch_normalization_5_w27 core=ROM_1P
#pragma HLS RESOURCE variable=batch_normalization_5_w28 core=ROM_1P
#pragma HLS RESOURCE variable=batch_normalization_5_w29 core=ROM_1P

#pragma HLS RESOURCE variable=depthwise_conv2d_3_w30 core=ROM_1P
#pragma HLS RESOURCE variable=batch_normalization_6_w31 core=ROM_1P
#pragma HLS RESOURCE variable=batch_normalization_6_w32 core=ROM_1P
#pragma HLS RESOURCE variable=batch_normalization_6_w33 core=ROM_1P
#pragma HLS RESOURCE variable=batch_normalization_6_w34 core=ROM_1P
#pragma HLS RESOURCE variable=conv2d_3_w35 core=ROM_1P
#pragma HLS RESOURCE variable=batch_normalization_7_w36 core=ROM_1P
#pragma HLS RESOURCE variable=batch_normalization_7_w37 core=ROM_1P
#pragma HLS RESOURCE variable=batch_normalization_7_w38 core=ROM_1P
#pragma HLS RESOURCE variable=batch_normalization_7_w39 core=ROM_1P

#pragma HLS RESOURCE variable=dense_w40 core=ROM_1P
#pragma HLS RESOURCE variable=dense_w41 core=ROM_1P
#pragma HLS RESOURCE variable=dense_1_w42 core=ROM_1P
#pragma HLS RESOURCE variable=dense_1_w43 core=ROM_1P

    // ===================== 1) Stream in image ================================
    unsigned int idx = 0;
    while (idx < img_words) {
    #pragma HLS PIPELINE II=1
        if (!s_axis.empty()) {
            axis_t pkt = s_axis.read();
            uint8_t px = (uint8_t)(pkt.data & 0xFF);   // one channel per word
            unsigned int i = idx / (IMG_SIZE * 3);
            unsigned int j = (idx / 3) % IMG_SIZE;
            unsigned int c = idx % 3;
            img[i][j][c] = px;
            idx++;
        }
    }
    debug_flag = 1;

    // ===================== 2) CNN compute ===================================
    // --- Block 1: DW + BN ---
    for (int c=0;c<3;c++)
      for (int i=0;i<IMG_SIZE;i++)
        for (int j=0;j<IMG_SIZE;j++){
            int32_t acc=0;
            for (int ki=0;ki<3;ki++)
              for (int kj=0;kj<3;kj++){
                int ii=i+ki-1; if(ii<0)ii=0; if(ii>=IMG_SIZE)ii=IMG_SIZE-1;
                int jj=j+kj-1; if(jj<0)jj=0; if(jj>=IMG_SIZE)jj=IMG_SIZE-1;
                int8_t w=depthwise_conv2d_w0[c*9+ki*3+kj];
                acc += ((int32_t)img[ii][jj][c]) * (int32_t)w;
              }
            dw1_out[i][j][c] = apply_bn(acc,
                batch_normalization_w1[c],
                batch_normalization_w2[c],
                batch_normalization_w3[c],
                batch_normalization_w4[c]);
        }
    // --- Block 1: PW + BN + ReLU ---
    for (int oc=0;oc<8;oc++)
      for (int i=0;i<IMG_SIZE;i++)
        for (int j=0;j<IMG_SIZE;j++){
            int32_t acc=0;
            for (int ic=0;ic<3;ic++){
                int8_t w=conv2d_w5[ic*8+oc];
                acc += ((int32_t)dw1_out[i][j][ic]) * (int32_t)w;
            }
            block1_pw[i][j][oc] = apply_bn_relu(acc,
                batch_normalization_1_w6[oc],
                batch_normalization_1_w7[oc],
                batch_normalization_1_w8[oc],
                batch_normalization_1_w9[oc]);
        }
    // --- Block 1: MaxPool 64->32 ---
    for (int oc=0;oc<8;oc++)
      for (int i=0;i<IMG_SIZE;i+=2)
        for (int j=0;j<IMG_SIZE;j+=2){
            int8_t w2[2][2]={{block1_pw[i][j][oc],block1_pw[i][j+1][oc]},
                             {block1_pw[i+1][j][oc],block1_pw[i+1][j+1][oc]}};
            block1_pool[i/2][j/2][oc] = maxpool2x2_int8(w2);
        }

    const int half = IMG_SIZE/2; // 32

    // --- Block 2: DW + BN ---
    for (int c=0;c<8;c++)
      for (int i=0;i<half;i++)
        for (int j=0;j<half;j++){
            int32_t acc=0;
            for (int ki=0;ki<3;ki++)
              for (int kj=0;kj<3;kj++){
                int ii=i+ki-1; if(ii<0)ii=0; if(ii>=half)ii=half-1;
                int jj=j+kj-1; if(jj<0)jj=0; if(jj>=half)jj=half-1;
                int8_t w=depthwise_conv2d_1_w10[c*9+ki*3+kj];
                acc += ((int32_t)block1_pool[ii][jj][c]) * (int32_t)w;
              }
            dw2_out[i][j][c] = apply_bn(acc,
                batch_normalization_2_w11[c],
                batch_normalization_2_w12[c],
                batch_normalization_2_w13[c],
                batch_normalization_2_w14[c]);
        }
    // --- Block 2: PW + BN + ReLU ---
    for (int oc=0;oc<16;oc++)
      for (int i=0;i<half;i++)
        for (int j=0;j<half;j++){
            int32_t acc=0;
            for (int ic=0;ic<8;ic++){
                int8_t w=conv2d_1_w15[ic*16+oc];
                acc += ((int32_t)dw2_out[i][j][ic]) * (int32_t)w;
            }
            block2_pw[i][j][oc] = apply_bn_relu(acc,
                batch_normalization_3_w16[oc],
                batch_normalization_3_w17[oc],
                batch_normalization_3_w18[oc],
                batch_normalization_3_w19[oc]);
        }
    // --- Block 2: MaxPool 32->16 ---
    for (int oc=0;oc<16;oc++)
      for (int i=0;i<half;i+=2)
        for (int j=0;j<half;j+=2){
            int8_t w2[2][2]={{block2_pw[i][j][oc],block2_pw[i][j+1][oc]},
                             {block2_pw[i+1][j][oc],block2_pw[i+1][j+1][oc]}};
            block2_pool[i/2][j/2][oc] = maxpool2x2_int8(w2);
        }

    const int quarter = half/2; // 16

    // --- Block 3: DW + BN ---
    for (int c=0;c<16;c++)
      for (int i=0;i<quarter;i++)
        for (int j=0;j<quarter;j++){
            int32_t acc=0;
            for (int ki=0;ki<3;ki++)
              for (int kj=0;kj<3;kj++){
                int ii=i+ki-1; if(ii<0)ii=0; if(ii>=quarter)ii=quarter-1;
                int jj=j+kj-1; if(jj<0)jj=0; if(jj>=quarter)jj=quarter-1;
                int8_t w=depthwise_conv2d_2_w20[c*9+ki*3+kj];
                acc += ((int32_t)block2_pool[ii][jj][c]) * (int32_t)w;
              }
            dw3_out[i][j][c] = apply_bn(acc,
                batch_normalization_4_w21[c],
                batch_normalization_4_w22[c],
                batch_normalization_4_w23[c],
                batch_normalization_4_w24[c]);
        }
    // --- Block 3: PW + BN + ReLU ---
    for (int oc=0;oc<32;oc++)
      for (int i=0;i<quarter;i++)
        for (int j=0;j<quarter;j++){
            int32_t acc=0;
            for (int ic=0;ic<16;ic++){
                int8_t w=conv2d_2_w25[ic*32+oc];
                acc += ((int32_t)dw3_out[i][j][ic]) * (int32_t)w;
            }
            block3_pw[i][j][oc] = apply_bn_relu(acc,
                batch_normalization_5_w26[oc],
                batch_normalization_5_w27[oc],
                batch_normalization_5_w28[oc],
                batch_normalization_5_w29[oc]);
        }
    // --- Block 3: MaxPool 16->8 ---
    for (int oc=0;oc<32;oc++)
      for (int i=0;i<quarter;i+=2)
        for (int j=0;j<quarter;j+=2){
            int8_t w2[2][2]={{block3_pw[i][j][oc],block3_pw[i][j+1][oc]},
                             {block3_pw[i+1][j][oc],block3_pw[i+1][j+1][oc]}};
            block3_pool[i/2][j/2][oc] = maxpool2x2_int8(w2);
        }

    const int eighth = quarter/2; // 8

    // --- Block 4: DW + BN ---
    for (int c=0;c<32;c++)
      for (int i=0;i<eighth;i++)
        for (int j=0;j<eighth;j++){
            int32_t acc=0;
            for (int ki=0;ki<3;ki++)
              for (int kj=0;kj<3;kj++){
                int ii=i+ki-1; if(ii<0)ii=0; if(ii>=eighth)ii=eighth-1;
                int jj=j+kj-1; if(jj<0)jj=0; if(jj>=eighth)jj=eighth-1;
                int8_t w=depthwise_conv2d_3_w30[c*9+ki*3+kj];
                acc += ((int32_t)block3_pool[ii][jj][c]) * (int32_t)w;
              }
            dw4_out[i][j][c] = apply_bn(acc,
                batch_normalization_6_w31[c],
                batch_normalization_6_w32[c],
                batch_normalization_6_w33[c],
                batch_normalization_6_w34[c]);
        }
    // --- Block 4: PW + BN + ReLU ---
    for (int oc=0;oc<32;oc++)
      for (int i=0;i<eighth;i++)
        for (int j=0;j<eighth;j++){
            int32_t acc=0;
            for (int ic=0;ic<32;ic++){
                int8_t w=conv2d_3_w35[ic*32+oc];
                acc += ((int32_t)dw4_out[i][j][ic]) * (int32_t)w;
            }
            block4_pw[i][j][oc] = apply_bn_relu(acc,
                batch_normalization_7_w36[oc],
                batch_normalization_7_w37[oc],
                batch_normalization_7_w38[oc],
                batch_normalization_7_w39[oc]);
        }
    // --- Block 4: MaxPool 8->4 (4x4x32 = 512) ---
    for (int oc=0;oc<32;oc++)
      for (int i=0;i<eighth;i+=2)
        for (int j=0;j<eighth;j+=2){
            int8_t w2[2][2]={{block4_pw[i][j][oc],block4_pw[i][j+1][oc]},
                             {block4_pw[i+1][j][oc],block4_pw[i+1][j+1][oc]}};
            block4_pool[i/2][j/2][oc] = maxpool2x2_int8(w2);
        }

    // --- Flatten 4x4x32 -> 512 ---
    int f=0;
    for (int i=0;i<4;i++)
      for (int j=0;j<4;j++)
        for (int c=0;c<32;c++)
            flat[f++] = block4_pool[i][j][c];

    // --- Dense1: 512 -> 32 (ReLU) ---
    for (int o=0;o<32;o++){
        int32_t acc=0;
        for (int i=0;i<512;i++)
            acc += ((int32_t)flat[i]) * (int32_t)dense_w40[i*32 + o];
        acc += ((int32_t)dense_w41[o]) << 7;   // bias scaled
        int32_t v = acc >> 7;
        fc1[o] = clamp_int8(v > 0 ? v : 0);
    }

    // --- Dense2: 32 -> 6 (logits) ---
    for (int o=0;o<N_CLASSES;o++){
        int32_t acc=0;
        for (int i=0;i<32;i++)
            acc += ((int32_t)fc1[i]) * (int32_t)dense_1_w42[i*N_CLASSES + o];
        acc += ((int32_t)dense_1_w43[o]) << 7; // bias scaled
        output[o] = clamp_int8(acc >> 7);
    }

    debug_flag = 2; // compute finished

    // ===================== 3) Stream out logits ==============================
    for (int o=0;o<N_CLASSES;o++){
    #pragma HLS PIPELINE II=1
        axis_t pkt;
        pkt.data = (uint32_t)( (uint8_t)output[o] );  // signed int8 in low 8 bits
        pkt.keep = -1;
        pkt.strb = 0;
        pkt.user = 0;
        pkt.last = (o==N_CLASSES-1) ? 1 : 0;
        m_axis.write(pkt);
    }
    debug_flag = 3; // done
    // === Drain a fixed number of remaining words to release DMA ===
    const int drain_max = 64 * 64 * 3;  // upper bound on possible inputs
    for (int k = 0; k < drain_max; k++) {
    #pragma HLS PIPELINE II=1
        if (!s_axis.empty()) {
            s_axis.read();
        } else {
            break;
        }
    }

    }
