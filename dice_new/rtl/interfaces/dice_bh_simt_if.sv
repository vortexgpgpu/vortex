interface dice_bh_simt_if
  import dice_pkg::*;
  import dice_frontend_pkg::*;
();

    logic                                          update_valid;
    logic                                          update_ready;
    dice_frontend_pkg::simt_stack_update_t         update_stack_data;
    logic [$clog2(DICE_NUM_MAX_CTA_PER_CORE)-1:0]  hw_cta_id;
    logic [1:0]                                    hw_cta_size;

    //branch handler (FDR)
    modport master (
        output update_valid,
        input  update_ready,
        output update_stack_data,
        output hw_cta_id,
        output hw_cta_size
    );

    //simt stack controller
    modport slave (
        input  update_valid,
        output update_ready,
        input  update_stack_data,
        input  hw_cta_id,
        input  hw_cta_size
    );

endinterface
