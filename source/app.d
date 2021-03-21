import std.exception;
import std.stdint;
import std.stdio;

void main()
{
    File f = File( "nn.bin", "r");

    uint32_t[] version_                  = f.rawRead( new uint32_t[1]);
    uint32_t[] hash                      = f.rawRead( new uint32_t[1]);
    uint32_t[] size                      = f.rawRead( new uint32_t[1]);
    char[]     architecture              = f.rawRead( new char[size[0]]); // char[178]
    uint32_t[] header1                   = f.rawRead( new uint32_t[1]);
    int16_t[]  featureTransformerBiases  = f.rawRead( new int16_t[256]);
    int16_t[]  featureTransformerWeights = f.rawRead( new int16_t[256 * 81* 1548]);
    uint32_t[] header2                   = f.rawRead( new uint32_t[1]);
    int32_t[]  hiddenLayer1Biases        = f.rawRead( new int32_t[32]);
    int8_t[]   hiddenLayer1Weights       = f.rawRead( new int8_t[32 * 512]);
    int32_t[]  hiddenLayer2Biases        = f.rawRead( new int32_t[32]);
    int8_t[]   hiddenLayer2Weights       = f.rawRead( new int8_t[32 * 32]);
    int32_t[]  outputLayerBiases         = f.rawRead( new int32_t[1]);
    int8_t[]   outputLayerWeights        = f.rawRead( new int8_t[1 * 32]);
    enforce( f.tell == f.size, "invalid file");

    writefln( "version: %#08x", version_[0]);    // => "version: 0x7af32f16"
    writefln( "hash   : %#08x", hash[0]);        // => "hash   : 0x3e5aa6ee"
    writefln( "architecture: %s", architecture); // => "architecture: Features=HalfKP(Friend)[125388->256x2],Network=AffineTransform[1<-32](ClippedReLU[32](AffineTransform[32<-32](ClippedReLU[32](AffineTransform[32<-512](InputSlice[512(0:512)])))))"
    writefln( "header1: %#08x", header1[0]);     // => "header1: 0x5d69d7b8"
    writefln( "header2: %#08x", header2[0]);     // => "header2: 0x63337156"
}
