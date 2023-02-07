package ahb_mux_pkg;

    typedef struct packed {
        logic      [31:0] HADDR;
        logic      [ 2:0] HBURST;
        logic             HMASTLOCK;
        logic      [ 2:0] HSIZE;
        logic      [ 1:0] HTRANS;
        logic             HWRITE;
    } aphase_t;

endpackage

