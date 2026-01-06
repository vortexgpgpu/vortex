//need to figure out how the active mask will be packaged
module decode
  import dice_pkg::*;
  import dice_frontend_pkg::*;
(
    input pgraph_meta_t metadata_i,
    input logic                            meta_in_valid_i,

    input thread_mask_t real_active_thread_mask_i,

    //To bitstream Fetcher
    output logic [DICE_ADDR_WIDTH-1:0]            bitstream_addr_o,
    output logic                                            bitstream_addr_valid_o,
    output logic [BITSTREAM_LENGTH_WIDTH-1:0] bitstream_length_o,

    //To branch handler
    output branch_meta_t branch_metadata_o,
    output logic                            branch_req_valid_o,

    //To valid checker
    output logic                       is_barrier_o,
    output fdr_meta_t meta_o
);

  // Bitstream Fetch
  assign bitstream_addr_o       = metadata_i.bitstream_addr;
  assign bitstream_length_o     = metadata_i.bitstream_length;
  assign bitstream_addr_valid_o = meta_in_valid_i;

  // Branch Handler
  assign branch_metadata_o      = metadata_i.branch_meta;
  assign branch_req_valid_o     = meta_in_valid_i;


  // valid checker
  assign is_barrier_o           = metadata_i.barrier;


  always_comb begin
    meta_o.bitstream_length = metadata_i.bitstream_length;
    meta_o.in_regs_bitmap   = metadata_i.in_regs_bitmap;
    meta_o.out_regs_bitmap  = metadata_i.out_regs_bitmap;
    meta_o.ld_dest_regs     = metadata_i.ld_dest_regs;
    meta_o.num_stores       = metadata_i.num_stores;
    meta_o.unrolling_factor = metadata_i.unrolling_factor;
    meta_o.lat              = metadata_i.lat;
    meta_o.parameter_load   = metadata_i.parameter_load;
  end

endmodule
