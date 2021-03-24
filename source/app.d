import misc;
import std.algorithm;
import std.exception;
import std.stdint;
import std.stdio;


void main()
{
    File f = File("nn.bin", "r");

    uint32_t[] version_                  = f.rawRead(new uint32_t[1]);
    uint32_t[] hash                      = f.rawRead(new uint32_t[1]);
    uint32_t[] size                      = f.rawRead(new uint32_t[1]);
    char[]     architecture              = f.rawRead(new char[size[0]]); // char[178]
    uint32_t[] header1                   = f.rawRead(new uint32_t[1]);
    int16_t[]  featureTransformerBiases  = f.rawRead(new int16_t[256]);
    int16_t[]  featureTransformerWeights = f.rawRead(new int16_t[256 * 81 * 1548]);
    uint32_t[] header2                   = f.rawRead(new uint32_t[1]);
    int32_t[]  hiddenLayer1Biases        = f.rawRead(new int32_t[32]);
    int8_t[]   hiddenLayer1Weights       = f.rawRead(new int8_t[32 * 512]);
    int32_t[]  hiddenLayer2Biases        = f.rawRead(new int32_t[32]);
    int8_t[]   hiddenLayer2Weights       = f.rawRead(new int8_t[32 * 32]);
    int32_t[]  outputLayerBiases         = f.rawRead(new int32_t[1]);
    int8_t[]   outputLayerWeights        = f.rawRead(new int8_t[1 * 32]);
    enforce(f.tell == f.size, "invalid file");

    writefln("version: %#08x", version_[0]);    // => "version: 0x7af32f16"
    writefln("hash   : %#08x", hash[0]);        // => "hash   : 0x3e5aa6ee"
    writefln("architecture: %s", architecture); // => "architecture: Features=HalfKP(Friend)[125388->256x2],Network=AffineTransform[1<-32](ClippedReLU[32](AffineTransform[32<-32](ClippedReLU[32](AffineTransform[32<-512](InputSlice[512(0:512)])))))"
    writefln("header1: %#08x", header1[0]);     // => "header1: 0x5d69d7b8"
    writefln("header2: %#08x", header2[0]);     // => "header2: 0x63337156"


    //foreach(i, e; outputLayerWeights) {
    //    writefln("%d:%d",i,e);
    //}


    //Piece a = {Color.WHITE, Type.LANCE, -1};
    //Position pos = parseSfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b -");
    Position pos = parseSfen("8l/1l+R2P3/p2pBG1pp/kps1p4/Nn1P2G2/P1P1P2PP/1PS6/1KSG3+r1/LN2+p3L w Sbgn3p 124");
    //Position pos = parseSfen("8l/1l+R2P3/p2pBG1pp/kps1p4/Nn1P2G2/P1P1P2PP/1PS6/1KSG3+r1/LN2+p3L w SBGN3P 124");
    //Position pos = parseSfen("lnsgkgsnl/1r5b1/p1ppppppp/7P1/1p7/9/PPPPPPP1P/1B5R1/LNSGKGSNL w -");
    //Position pos = parseSfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/7P1/PPPPPPP1P/1B5R1/LNSGKGSNL w -");
    writeln(pos.toKi2());

    int[38] blist, wlist;
    mklist(pos, blist, wlist);

    int16_t[256] a;
    int16_t[256] b;
    w1(blist.ptr, featureTransformerBiases.ptr, featureTransformerWeights.ptr, a.ptr);
    w1(wlist.ptr, featureTransformerBiases.ptr, featureTransformerWeights.ptr, b.ptr);

    uint8_t[512] c;
    transform(b.ptr, a.ptr, c.ptr);

    uint8_t[32] d;
    w2(c.ptr, hiddenLayer1Biases.ptr, hiddenLayer1Weights.ptr, d.ptr);

    uint8_t[32] e;
    w3(d.ptr, hiddenLayer2Biases.ptr, hiddenLayer2Weights.ptr, e.ptr);

    int32_t[1] f0;
    w4(e.ptr, outputLayerBiases.ptr, outputLayerWeights.ptr, f0.ptr);

    foreach (i; f0) {
        writeln(i / 16);
    }
}


uint8_t clippedReLU(int32_t input) {
    return cast(uint8_t)(max(0, min(127, input >> 6)));
}

void transform(int16_t* p0, int16_t* p1, uint8_t* output) {
    for (int i = 0; i < 256; i++) {
        output[0 * 256 + i] = cast(uint8_t)(max(0, min(127, p0[i])));
    }
    for (int i = 0; i < 256; i++) {
        output[1 * 256 + i] = cast(uint8_t)(max(0, min(127, p1[i])));
    }
}

void w1(int* kp, int16_t* biases, int16_t* weights, int16_t* output) {
    for (int i = 0; i < 256; i++) { // 行
        output[i] = biases[i];
    }
    for (int j = 0; j < 38; j++) { // 列
        for (int i = 0; i < 256; i++) { // 行
            output[i] += weights[kp[j] * 256 + i];
        }
    }
}

// ClippedReLU[32](AffineTransform[32<-512])
void w2(uint8_t* input, int32_t* biases, int8_t* weights, uint8_t* output) {
    for (int i = 0; i < 32; i++) { // 行
        int32_t sum = biases[i];
        for (int j = 0; j < 512; j++) { // 列
            sum += weights[i * 512 + j] * input[j];
        }
        output[i] = clippedReLU(sum);
    }
}

// ClippedReLU[32](AffineTransform[32<-32])
void w3(uint8_t* input, int32_t* biases, int8_t* weights, uint8_t* output) {
    for (int i = 0; i < 32; i++) { // 行
        int32_t sum = biases[i];
        for (int j = 0; j < 32; j++) { // 列
            sum += weights[i * 32 + j] * input[j];
        }
        output[i] = clippedReLU(sum);
    }
}

// AffineTransform[1<-32]
void w4(uint8_t* input, int32_t* biases, int8_t* weights, int32_t* output) {
    for (int i = 0; i < 1; i++) { // 行
        int32_t sum = biases[i];
        for (int j = 0; j < 32; j++) { // 列
            sum += weights[i * 32 + j] * input[j];
        }
        output[i] = sum;
    }
}
