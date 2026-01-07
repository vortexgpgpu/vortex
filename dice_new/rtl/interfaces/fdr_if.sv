
interface fdr_if import dice_frontend_pkg::*; ();

    logic valid;
    fdr_t data;
    logic ready;

    modport master (
        output valid,
        output data,
        input  ready
    );

    modport slave (
        input  valid,
        input  data,
        output ready
    );

endinterface
