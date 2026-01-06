interface branch_handler_if import dice_pkg::*; ();

    branch_predict_interface_t bh_data;
    dice_cta_status_t [DICE_NUM_MAX_CTA_PER_CORE-1:0] cta_status_data;
    logic              branch_predict_info_write_enable;

    //branch handler
    modport master (
        output bh_data,
        output branch_predict_info_write_enable,
        input cta_status_data
    );

    //status table
    modport slave (
        input bh_data,
        input branch_predict_info_write_enable,
        output cta_status_data
    );

endinterface






