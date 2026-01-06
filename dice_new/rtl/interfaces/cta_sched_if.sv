
interface cta_sched_if import dice_frontend_pkg::*; ();

    logic       valid;
    dice_frontend_pkg::schedule_eblock_t data;
    logic       ready;

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






