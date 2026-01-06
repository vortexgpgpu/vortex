interface dice_bh_simt_if import dice_frontend_pkg::*; ();
    logic                   update_valid;
    logic                   update_ready;
    dice_frontend_pkg::simt_stack_update_t     update_stack_data;

    //branch handler
    modport master (
        output update_valid,
        input  update_ready,
        output update_stack_data
    );

    //simt stack controller
    modport slave (
        input update_valid,
        output update_ready,
        input update_stack_data
    );

endinterface
