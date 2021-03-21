module misc;

import std.algorithm;
import std.array;
import std.ascii;
import std.conv;
import std.format;
import std.regex;
import std.stdint;
import std.stdio;
import std.string;
import std.typecons;


/**
 * 駒の色
 */
enum Color
{
    BLACK             = 0,  // 先手
    WHITE             = 1,  // 後手
    NONE              = 2,  // なし
}

/**
 * 駒の種類
 */
enum Type
{
    PAWN              = 0,  // 歩
    LANCE             = 1,  // 香
    KNIGHT            = 2,  // 桂
    SILVER            = 3,  // 銀
    GOLD              = 4,  // 金
    BISHOP            = 5,  // 角
    ROOK              = 6,  // 飛
    KING              = 7,  // 王
    PROMOTED_PAWN     = 8,  // と
    PROMOTED_LANCE    = 9,  // 成香
    PROMOTED_KNIGHT   = 10, // 成桂
    PROMOTED_SILVER   = 11, // 成銀
    PROMOTED_BISHOP   = 12, // 馬
    PROMOTED_ROOK     = 13, // 龍
    EMPTY             = 14, // 空
}

/**
 * 駒
 *
 * address:
 *  9  8  7  6  5  4  3  2  1
 * --------------------------+
 * 72 63 54 45 36 27 18  9  0|一
 * 73 64 55 46 37 28 19 10  1|二
 * 74 65 56 47 38 29 20 11  2|三
 * 75 66 57 48 39 30 21 12  3|四
 * 76 67 58 49 40 31 22 13  4|五
 * 77 68 59 50 41 32 23 14  5|六
 * 78 69 60 51 42 33 24 15  6|七
 * 79 70 61 52 43 34 25 16  7|八
 * 80 71 62 53 44 35 26 17  8|九
 */
struct Piece {
    Color color;
    Type type;
    int address;
}

/**
 * 局面
 *
 * 駒が40枚あるはず
 */
struct Position
{
    Piece[40] pieces;
    Color sideToMove;
}

/**
 * SFENをパースして局面を作って返す
 *
 * 例: 8l/1l+R2P3/p2pBG1pp/kps1p4/Nn1P2G2/P1P1P2PP/1PS6/1KSG3+r1/LN2+p3L w Sbgn3p 124
 */
Position parseSfen(string sfen)
{
    immutable Tuple!(Color, Type)[string] COLOR_TYPE = [
        "1":  tuple(Color.NONE, Type.EMPTY),
        "P":  tuple(Color.BLACK, Type.PAWN),
        "L":  tuple(Color.BLACK, Type.LANCE),
        "N":  tuple(Color.BLACK, Type.KNIGHT),
        "S":  tuple(Color.BLACK, Type.SILVER),
        "G":  tuple(Color.BLACK, Type.GOLD),
        "B":  tuple(Color.BLACK, Type.BISHOP),
        "R":  tuple(Color.BLACK, Type.ROOK),
        "K":  tuple(Color.BLACK, Type.KING),
        "+P": tuple(Color.BLACK, Type.PROMOTED_PAWN),
        "+L": tuple(Color.BLACK, Type.PROMOTED_LANCE),
        "+N": tuple(Color.BLACK, Type.PROMOTED_KNIGHT),
        "+S": tuple(Color.BLACK, Type.PROMOTED_SILVER),
        "+B": tuple(Color.BLACK, Type.PROMOTED_BISHOP),
        "+R": tuple(Color.BLACK, Type.PROMOTED_ROOK),
        "p":  tuple(Color.WHITE, Type.PAWN),
        "l":  tuple(Color.WHITE, Type.LANCE),
        "n":  tuple(Color.WHITE, Type.KNIGHT),
        "s":  tuple(Color.WHITE, Type.SILVER),
        "g":  tuple(Color.WHITE, Type.GOLD),
        "b":  tuple(Color.WHITE, Type.BISHOP),
        "r":  tuple(Color.WHITE, Type.ROOK),
        "k":  tuple(Color.WHITE, Type.KING),
        "+p": tuple(Color.WHITE, Type.PROMOTED_PAWN),
        "+l": tuple(Color.WHITE, Type.PROMOTED_LANCE),
        "+n": tuple(Color.WHITE, Type.PROMOTED_KNIGHT),
        "+s": tuple(Color.WHITE, Type.PROMOTED_SILVER),
        "+b": tuple(Color.WHITE, Type.PROMOTED_BISHOP),
        "+r": tuple(Color.WHITE, Type.PROMOTED_ROOK),
    ];

    Position pos;
    int index = 0;

    string[] a = sfen.split(" ");
    string boardState = a[0];
    string sideToMove = a[1];
    string piecesInHand = a[2];

    // 手番
    if (sideToMove != "b" && sideToMove != "w") {
        throw new StringException(sfen);
    }
    pos.sideToMove = sideToMove == "b" ? Color.BLACK : Color.WHITE;

    // 盤面
    for (int i = 9; i >= 2; i--) {
        boardState = boardState.replace(to!string(i), "1".replicate(i)); // 2～9を1に開いておく
    }
    boardState = boardState.replace("/", "");
    auto m = boardState.matchAll(r"\+?.");
    for (int rank = 0; rank <= 8; rank++) {
        for (int file = 8; file >= 0; file--) {
            auto t = COLOR_TYPE[m.front.hit];
            if (t[1] != Type.EMPTY) {
                pos.pieces[index++] = Piece(t[0], t[1], file * 9 + rank);
            }
            m.popFront();
        }
    }

    // 持ち駒
    if (piecesInHand != "-") {
        // 例：S, 4P, b, 3n, p, 18P
        foreach (c; piecesInHand.matchAll(r"(\d*)(\D)")) {
            int n = c[1] == "" ? 1 : to!int(c[1]);
            auto t = COLOR_TYPE[c[2]];
            foreach (i; 0 .. n) {
                pos.pieces[index++] = Piece(t[0], t[1], -1); // 持ち駒はアドレス:-1にしておく
            }
        }
    }

    return pos;
}

Piece lookAt(Position pos, int address)
{
    foreach(p; pos.pieces) {
        if (p.address == address) return p;
    }
    return Piece(Color.NONE, Type.EMPTY, -1);
}

string toKi2(ref Position pos)
{
    immutable string[] COLOR_STR = [" ", "v", " "];
    immutable string[] TYPE_STR = ["歩", "香", "桂", "銀", "金", "角", "飛", "玉", "と", "杏", "圭", "全", "馬", "龍", "・"];
    immutable string[] NUM_STR = ["〇", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "十一", "十二", "十三", "十四", "十五", "十六", "十七", "十八"];

    string hand(ref Position pos, Color color)
    {
        int[7] num; // 0:歩 1:香 2:桂 3:銀 4:金 5:角 6:飛
        foreach (Piece p; pos.pieces) {
            if (p.address == -1 && p.color == color) num[p.type] += 1;
        }
        string s;
        foreach_reverse (i, n; num) {
            if (n > 0) s ~= TYPE_STR[i];
            if (n > 1) s ~= NUM_STR[n];   
        }
        if (s.length == 0) s = "なし";
        return s;
    }

    string s;
    s ~= format("後手の持駒：%s\n", hand(pos, Color.WHITE));
    s ~= "  ９ ８ ７ ６ ５ ４ ３ ２ １\n";
    s ~= "+---------------------------+\n";
    for (int rank = 0; rank <= 8; rank++) {
        s ~= "|";
        for (int file = 8; file >= 0; file--) {
            Piece p = pos.lookAt(file * 9 + rank);
            s ~= COLOR_STR[p.color] ~ TYPE_STR[p.type];
        }
        s ~= format("|%s\n", NUM_STR[rank + 1]);
    }
    s ~= "+---------------------------+\n";
    s ~= format("先手の持駒：%s\n", hand(pos, Color.BLACK));
    return s;
}

void mklist(ref Position pos, out int[38] blist, out int[38] wlist)
{
    /*
     * 持ち駒（手番と駒のタイプ）からオフセットを引く表
     * key: [color_t][type_t]
     * see https://github.com/HiraokaTakuya/apery/blob/32216277e51c3b008e3c8eea6954f1bb3c416b57/src/evaluate.hpp#L36
     */
    immutable short[][] OFFSET_HAND = [
      // 歩, 香, 桂, 銀, 金, 角, 飛,
        [ 1, 39, 49, 59, 69, 79, 85],
        [20, 44, 54, 64, 74, 82, 88],
    ];

    /*
     * 駒からオフセットを引く表
     * key: Square
     * see https://github.com/HiraokaTakuya/apery/blob/32216277e51c3b008e3c8eea6954f1bb3c416b57/src/evaluate.hpp#L36
     */
    immutable short[][] OFFSET_PP = [
       // 歩,  香,  桂,  銀,  金,  角,   飛, 王,  と, 成香, 成桂, 成銀,   馬,   龍,
        [ 90, 252, 414, 576, 738, 900, 1224,  0, 738,  738,  738,  738, 1062, 1386],
        [171, 333, 495, 657, 819, 981, 1305,  0, 819,  819,  819,  819, 1143, 1467],
    ];

    int bk, wk, index;

    foreach (p; pos.pieces) {
        if (p.color == Color.BLACK && p.type == Type.KING) bk =      p.address;
        if (p.color == Color.WHITE && p.type == Type.KING) wk = 80 - p.address;
    }

    int[7][2] num; // 0:歩 1:香 2:桂 3:銀 4:金 5:角 6:飛
    foreach (p; pos.pieces) {
        if (p.address == -1) num[p.color][p.type] += 1;
    }
    foreach (color; 0..2) {
        foreach (i, n; num[color]) {
            foreach (j; 0..n) {
                blist[index] = bk * 1548 + OFFSET_HAND[color    ][i] + j;
                wlist[index] = wk * 1548 + OFFSET_HAND[color ^ 1][i] + j;
                index += 1;
            }
        }
    }

    foreach (p; pos.pieces) {
        if (p.address >= 0 && p.type != Type.KING) {
            blist[index] = bk * 1548 + OFFSET_PP[p.color    ][p.type] +       p.address;
            wlist[index] = wk * 1548 + OFFSET_PP[p.color ^ 1][p.type] + (80 - p.address);
            index += 1;
        }
    }

    sort(blist[]);
    sort(wlist[]);
}
