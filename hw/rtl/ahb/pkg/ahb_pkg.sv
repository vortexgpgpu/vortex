
package ahb_pkg;

    typedef enum logic [1:0] {
        IDLE    = 2'b00,
        SEQ     = 2'b01,
        NONSEQ  = 2'b10,
        BUSY    = 2'b11
    } HTRANS_t;

    typedef enum logic [2:0] {
        SINGLE  = 3'b000,
        INCR    = 3'b001,
        WRAP_4  = 3'b010,
        INCR_4  = 3'b011,
        WRAP_8  = 3'b100,
        INCR_8  = 3'b101,
        WRAP_16 = 3'b110,
        INCR_16 = 3'b111
    } HBURST_t;

    typedef enum logic [2:0] {
        BYTE            = 3'b000,
        HALFWORD        = 3'b001,
        WORD            = 3'b010,
        DOUBLEWORD      = 3'b011,
        FOUR_WORD_LINE  = 3'b100,
        EIGHT_WORD_LINE = 3'b101
    } HSIZE_t;

endpackage
