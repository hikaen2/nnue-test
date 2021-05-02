import misc;
import std.algorithm;
import std.exception;
import std.stdint;
import std.stdio;

/**
 * 入力層
 */
struct InputSlice(int outputDimensions) {
    alias OutputType = uint8_t;
    static immutable OUTPUT_DIMENSIONS = outputDimensions;

    void readParameters(File f) {
        return;
    }

    uint8_t* propagate(uint8_t* transformed_features, uint8_t* buffer) {
        return transformed_features;
    }
}

/**
 * Clipped ReLU
 */
struct ClippedReLU(PreviousLayer) {
    alias InputType = PreviousLayer.OutputType;
    alias OutputType = uint8_t;
    static immutable INPUT_DIMENSIONS = PreviousLayer.OUTPUT_DIMENSIONS;
    static immutable OUTPUT_DIMENSIONS = PreviousLayer.OUTPUT_DIMENSIONS;
    static immutable SELF_BUFFER_SIZE = 0; // TODO
    static immutable WEIGHT_SCALE_BITS = 6;

    PreviousLayer previousLayer;

    void readParameters(File f) {
        previousLayer.readParameters(f);
    }

    OutputType* propagate(uint8_t* transformed_features, uint8_t* buffer) {
        auto input = previousLayer.propagate(transformed_features, buffer + SELF_BUFFER_SIZE);
        auto output = cast(OutputType*)buffer;
        for (int i = 0; i < INPUT_DIMENSIONS; i++) {
            output[i] = cast(OutputType)(max(0, min(127, input[i] >> WEIGHT_SCALE_BITS)));
        }
        return output;
    }
}

/**
 * アフィン変換層
 */
struct AffineTransform(PreviousLayer, int outputDimensions) {
    alias InputType = PreviousLayer.OutputType;
    alias OutputType = int32_t;
    static immutable INPUT_DIMENSIONS = PreviousLayer.OUTPUT_DIMENSIONS;
    static immutable OUTPUT_DIMENSIONS = outputDimensions;
    static immutable SELF_BUFFER_SIZE = 0; // TODO

    PreviousLayer previousLayer;
    OutputType[OUTPUT_DIMENSIONS] biases;
    int8_t[OUTPUT_DIMENSIONS * INPUT_DIMENSIONS] weights;

    void readParameters(File f) {
        previousLayer.readParameters(f);
        f.rawRead(biases);
        f.rawRead(weights);
    }

    OutputType* propagate(uint8_t* transformed_features, uint8_t* buffer) {
        auto input = previousLayer.propagate(transformed_features, buffer + SELF_BUFFER_SIZE);
        auto output = cast(OutputType*)buffer;

        for (int i = 0; i < OUTPUT_DIMENSIONS; i++) {
            OutputType sum = biases[i];
            for (int j = 0; j < INPUT_DIMENSIONS; j++) {
                sum += weights[i * INPUT_DIMENSIONS + j] * input[j];
            }
            output[i] = sum;
        }

        return output;
    }
}

/**
 * 入力特徴量変換器
 */
struct FeatureTransformer {
    alias OutputType = uint8_t;
    static immutable HALF_DIMENSIONS = 256;
    static immutable INPUT_DIMENSIONS = 81 * 1548;
    static immutable OUTPUT_DIMENSIONS = HALF_DIMENSIONS * 2;

    int16_t[HALF_DIMENSIONS] biases;
    int16_t[HALF_DIMENSIONS * INPUT_DIMENSIONS] weights;

    public void readParameters(File f) {
        f.rawRead(biases);
        f.rawRead(weights);
    }
}

alias InputLayer = InputSlice!(512);
alias HiddenLayer1 = ClippedReLU!(AffineTransform!(InputLayer, 32));
alias HiddenLayer2 = ClippedReLU!(AffineTransform!(HiddenLayer1, 32));
alias OutputLayer = AffineTransform!(HiddenLayer2, 1);
alias Network = OutputLayer;

void main()
{
    FeatureTransformer* featureTransformer = new FeatureTransformer();
    Network* network = new Network();

    File f = File("nn.bin", "r");
    uint32_t ver = f.rawRead(new uint32_t[1])[0];
    uint32_t hash = f.rawRead(new uint32_t[1])[0];
    uint32_t size = f.rawRead(new uint32_t[1])[0];
    char[] arch = f.rawRead(new char[size]);
    uint32_t header1 = f.rawRead(new uint32_t[1])[0];
    featureTransformer.readParameters(f);
    uint32_t header2 = f.rawRead(new uint32_t[1])[0];
    network.readParameters(f);



    // uint32_t[] version_                  = f.rawRead(new uint32_t[1]);
    // uint32_t[] hash                      = f.rawRead(new uint32_t[1]);
    // uint32_t[] size                      = f.rawRead(new uint32_t[1]);
    // char[]     architecture              = f.rawRead(new char[size[0]]); // char[178]
    // uint32_t[] header1                   = f.rawRead(new uint32_t[1]);
    // int16_t[]  featureTransformerBiases  = f.rawRead(new int16_t[256]);
    // int16_t[]  featureTransformerWeights = f.rawRead(new int16_t[256 * 81 * 1548]);
    // uint32_t[] header2                   = f.rawRead(new uint32_t[1]);
    // int32_t[]  hiddenLayer1Biases        = f.rawRead(new int32_t[32]);
    // int8_t[]   hiddenLayer1Weights       = f.rawRead(new int8_t[32 * 512]);
    // int32_t[]  hiddenLayer2Biases        = f.rawRead(new int32_t[32]);
    // int8_t[]   hiddenLayer2Weights       = f.rawRead(new int8_t[32 * 32]);
    // int32_t[]  outputLayerBiases         = f.rawRead(new int32_t[1]);
    // int8_t[]   outputLayerWeights        = f.rawRead(new int8_t[1 * 32]);
    // enforce(f.tell == f.size, "invalid loading");

    // writefln("version: %#08x", version_[0]);    // => "version: 0x7af32f16"
    // writefln("hash   : %#08x", hash[0]);        // => "hash   : 0x3e5aa6ee"
    // writefln("architecture: %s", architecture); // => "architecture: Features=HalfKP(Friend)[125388->256x2],Network=AffineTransform[1<-32](ClippedReLU[32](AffineTransform[32<-32](ClippedReLU[32](AffineTransform[32<-512](InputSlice[512(0:512)])))))"
    // writefln("header1: %#08x", header1[0]);     // => "header1: 0x5d69d7b8"
    // writefln("header2: %#08x", header2[0]);     // => "header2: 0x63337156"


    // //foreach(i, e; outputLayerWeights) {
    // //    writefln("%d:%d", i, e);
    // //}


    // //Position pos = parseSfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b -");
    // Position pos = parseSfen("8l/1l+R2P3/p2pBG1pp/kps1p4/Nn1P2G2/P1P1P2PP/1PS6/1KSG3+r1/LN2+p3L w Sbgn3p 124");
    // //Position pos = parseSfen("8l/1l+R2P3/p2pBG1pp/kps1p4/Nn1P2G2/P1P1P2PP/1PS6/1KSG3+r1/LN2+p3L w SBGN3P 124");
    // //Position pos = parseSfen("lnsgkgsnl/1r5b1/p1ppppppp/7P1/1p7/9/PPPPPPP1P/1B5R1/LNSGKGSNL w -");
    // //Position pos = parseSfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/7P1/PPPPPPP1P/1B5R1/LNSGKGSNL w -");
    // //Position pos = parseSfen("8K/8P/9/9/9/9/9/9/k8 b -");
    // writeln(pos.toKi2());


    // int[38] blist, wlist;
    // mklist(pos, blist, wlist); // 特徴ベクトルを作る

    // int16_t[256] yb;
    // int16_t[256] yw;
    // w1(blist.ptr, featureTransformerBiases.ptr, featureTransformerWeights.ptr, yb.ptr); // blistからybを作る（先手の分）
    // w1(wlist.ptr, featureTransformerBiases.ptr, featureTransformerWeights.ptr, yw.ptr); // wlistからybを作る（後手の分）

    // uint8_t[512] z1;
    // if (pos.sideToMove == Color.BLACK) {
    //     transform(yb.ptr, yw.ptr, z1.ptr); // yb, ywを連結してz1を作る（先手番）
    // } else {
    //     transform(yw.ptr, yb.ptr, z1.ptr); // yw, ybを連結してz1を作る（後手番）
    // }

    // uint8_t[32] z2;
    // w2(z1.ptr, hiddenLayer1Biases.ptr, hiddenLayer1Weights.ptr, z2.ptr); // z1からz2を作る

    // uint8_t[32] z3;
    // w3(z2.ptr, hiddenLayer2Biases.ptr, hiddenLayer2Weights.ptr, z3.ptr); // z2からz3を作る

    // int32_t[1] z4;
    // w4(z3.ptr, outputLayerBiases.ptr, outputLayerWeights.ptr, z4.ptr); // z3からz4を作る

    // writeln(z4[0] / 16);
}


// void transform(int16_t* p0, int16_t* p1, uint8_t* output) {
//     for (int i = 0; i < 256; i++) {
//         output[0 * 256 + i] = cast(uint8_t)(max(0, min(127, p0[i])));
//     }
//     for (int i = 0; i < 256; i++) {
//         output[1 * 256 + i] = cast(uint8_t)(max(0, min(127, p1[i])));
//     }
// }

// void w1(int* kp, int16_t* biases, int16_t* weights, int16_t* output) {
//     for (int i = 0; i < 256; i++) { // 行
//         output[i] = biases[i];
//     }
//     for (int j = 0; j < 38; j++) { // 列
//         for (int i = 0; i < 256; i++) { // 行
//             output[i] += weights[kp[j] * 256 + i];
//         }
//     }
// }

// // ClippedReLU[32](AffineTransform[32<-512])
// void w2(uint8_t* input, int32_t* biases, int8_t* weights, uint8_t* output) {
//     for (int i = 0; i < 32; i++) { // 行
//         int32_t sum = biases[i];
//         for (int j = 0; j < 512; j++) { // 列
//             sum += weights[i * 512 + j] * input[j];
//         }
//         output[i] = cast(uint8_t)(max(0, min(127, sum >> 6)));
//     }
// }

// // ClippedReLU[32](AffineTransform[32<-32])
// void w3(uint8_t* input, int32_t* biases, int8_t* weights, uint8_t* output) {
//     for (int i = 0; i < 32; i++) { // 行
//         int32_t sum = biases[i];
//         for (int j = 0; j < 32; j++) { // 列
//             sum += weights[i * 32 + j] * input[j];
//         }
//         output[i] = cast(uint8_t)(max(0, min(127, sum >> 6)));
//     }
// }

// // AffineTransform[1<-32]
// void w4(uint8_t* input, int32_t* biases, int8_t* weights, int32_t* output) {
//     for (int i = 0; i < 1; i++) { // 行
//         int32_t sum = biases[i];
//         for (int j = 0; j < 32; j++) { // 列
//             sum += weights[i * 32 + j] * input[j];
//         }
//         output[i] = sum;
//     }
// }
