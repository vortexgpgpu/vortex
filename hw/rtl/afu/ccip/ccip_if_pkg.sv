// Date: 02/2/2016
// Compliant with CCI-P spec v0.71
package ccip_if_pkg;

//=====================================================================
// CCI-P interface defines
//=====================================================================
parameter CCIP_VERSION_NUMBER    = 12'h071;

parameter CCIP_CLADDR_WIDTH      = 42; 
parameter CCIP_CLDATA_WIDTH      = 512; 

parameter CCIP_MMIOADDR_WIDTH    = 16; 
parameter CCIP_MMIODATA_WIDTH    = 64; 
parameter CCIP_TID_WIDTH         = 9; 

parameter CCIP_MDATA_WIDTH       = 16; 


// Number of requests that can be accepted after almost full is asserted. 
parameter CCIP_TX_ALMOST_FULL_THRESHOLD = 8; 

parameter CCIP_MMIO_RD_TIMEOUT = 512;

parameter CCIP_SYNC_RESET_POLARITY=1;       // Active High Reset

// Base types 
//---------------------------------------------------------------------- 
typedef logic [CCIP_CLADDR_WIDTH-1:0]   t_ccip_clAddr;
typedef logic [CCIP_CLDATA_WIDTH-1:0]   t_ccip_clData;


typedef logic [CCIP_MMIOADDR_WIDTH-1:0] t_ccip_mmioAddr;
typedef logic [CCIP_MMIODATA_WIDTH-1:0] t_ccip_mmioData;
typedef logic [CCIP_TID_WIDTH-1:0]      t_ccip_tid;


typedef logic [CCIP_MDATA_WIDTH-1:0]    t_ccip_mdata;
typedef logic [1:0]                     t_ccip_clNum;
typedef logic [2:0]                     t_ccip_qwIdx;

  
// Request Type  Encodings
//----------------------------------------------------------------------
// Channel 0
typedef enum logic [3:0] {
    eREQ_RDLINE_I  = 4'h0,      // Memory Read with FPGA Cache Hint=Invalid
    eREQ_RDLINE_S  = 4'h1       // Memory Read with FPGA Cache Hint=Shared
} t_ccip_c0_req;

// Channel 1
typedef enum logic [3:0] {
    eREQ_WRLINE_I  = 4'h0,      // Memory Write with FPGA Cache Hint=Invalid
    eREQ_WRLINE_M  = 4'h1,      // Memory Write with FPGA Cache Hint=Modified
    eREQ_WRPUSH_I  = 4'h2,      // Memory Write with DDIO Hint ** NOT SUPPORTED CURRENTLY **
    eREQ_WRFENCE   = 4'h4,      // Memory Write Fence
//    eREQ_ATOMIC    = 4'h5,      // Atomic operation: Compare-Exchange for Memory Addr  ** NOT SUPPORTED CURRENTELY **
    eREQ_INTR      = 4'h6       // Interrupt the CPU ** NOT SUPPORTED CURRENTLY **
} t_ccip_c1_req;

// Response Type  Encodings
//----------------------------------------------------------------------
// Channel 0
typedef enum logic [3:0] {
    eRSP_RDLINE     = 4'h0,     // Memory Read
    eRSP_UMSG       = 4'h4      // UMsg received
//    eRSP_ATOMIC     = 4'h5      // Atomic Operation: Compare-Exchange for Memory Addr
} t_ccip_c0_rsp;

// Channel 1
typedef enum logic [3:0] {
    eRSP_WRLINE     = 4'h0,     // Memory Write
    eRSP_WRFENCE    = 4'h4,     // Memory Write Fence 
    eRSP_INTR       = 4'h6      // Interrupt delivered to the CPU ** NOT SUPPORTED CURRENTLY **
} t_ccip_c1_rsp;

//
// Virtual Channel Select
//----------------------------------------------------------------------
typedef enum logic [1:0] {
    eVC_VA  = 2'b00,
    eVC_VL0 = 2'b01,
    eVC_VH0 = 2'b10,
    eVC_VH1 = 2'b11
} t_ccip_vc;

// Multi-CL Memory Request
//----------------------------------------------------------------------
typedef enum logic [1:0] {
    eCL_LEN_1 = 2'b00,
    eCL_LEN_2 = 2'b01,
    eCL_LEN_4 = 2'b11
} t_ccip_clLen;

//
// Structures for Request and Response headers
//----------------------------------------------------------------------
typedef struct packed {
    t_ccip_vc       vc_sel;
    logic [1:0]     rsvd1;     // reserved, drive 0
    t_ccip_clLen    cl_len;
    t_ccip_c0_req   req_type;
    logic [5:0]     rsvd0;     // reserved, drive 0
    t_ccip_clAddr   address;
    t_ccip_mdata    mdata;
} t_ccip_c0_ReqMemHdr;
parameter CCIP_C0TX_HDR_WIDTH = $bits(t_ccip_c0_ReqMemHdr);

typedef struct packed {
    logic [5:0]     rsvd2;
    t_ccip_vc       vc_sel;
    logic           sop;
    logic           rsvd1;     // reserved, drive 0
    t_ccip_clLen    cl_len;
    t_ccip_c1_req   req_type;
    logic [5:0]     rsvd0;     // reserved, drive 0
    t_ccip_clAddr   address;
    t_ccip_mdata    mdata;
} t_ccip_c1_ReqMemHdr;
parameter CCIP_C1TX_HDR_WIDTH = $bits(t_ccip_c1_ReqMemHdr);

typedef struct packed {
    logic [5:0]     rsvd2;          // reserved, drive 0
    t_ccip_vc       vc_sel;
    logic [3:0]     rsvd1;          // reserved, drive 0
    t_ccip_c1_req   req_type;
    logic [47:0]    rsvd0;          // reserved, drive 0
    t_ccip_mdata    mdata;
}t_ccip_c1_ReqFenceHdr;

typedef struct packed {
    t_ccip_vc       vc_used;
    logic           rsvd1;          // reserved, don't care
    logic           hit_miss;
    logic [1:0]     rsvd0;          // reserved, don't care
    t_ccip_clNum    cl_num;
    t_ccip_c0_rsp   resp_type;
    t_ccip_mdata    mdata;
} t_ccip_c0_RspMemHdr;
parameter CCIP_C0RX_HDR_WIDTH = $bits(t_ccip_c0_RspMemHdr);

typedef struct packed {
    t_ccip_vc       vc_used;
    logic           rsvd1;          // reserved, don't care
    logic           hit_miss;
    logic           format;
    logic           rsvd0;          // reserved, don't care
    t_ccip_clNum    cl_num;
    t_ccip_c1_rsp   resp_type;
    t_ccip_mdata    mdata;
} t_ccip_c1_RspMemHdr;
parameter CCIP_C1RX_HDR_WIDTH = $bits(t_ccip_c1_RspMemHdr);

typedef struct packed {
    logic [7:0]     rsvd0;          // reserved, don't care
    t_ccip_c1_rsp   resp_type;
    t_ccip_mdata    mdata;
} t_ccip_c1_RspFenceHdr;

// Alternate Channel 0 MMIO request from host : 
//  MMIO requests arrive on the same channel as read responses, sharing
//  t_if_ccip_c0_Rx below.  When either mmioRdValid or mmioWrValid is set
//  the message is an MMIO request and should be processed by casting
//  t_if_ccip_c0_Rx.hdr to t_ccip_c0_ReqMmioHdr.
typedef struct packed {
    t_ccip_mmioAddr address;    // 4B aligned Mmio address
    logic [1:0]     length;     // 2'b00- 4B, 2'b01- 8B, 2'b10- 64B
    logic           rsvd;       // reserved, don't care
    t_ccip_tid      tid;
} t_ccip_c0_ReqMmioHdr;

typedef struct packed {
    t_ccip_tid     tid;         // Returned back from ReqMmioHdr
} t_ccip_c2_RspMmioHdr;
parameter CCIP_C2TX_HDR_WIDTH = $bits(t_ccip_c2_RspMmioHdr);

//------------------------------------------------------------------------
// CCI-P Input & Output bus structures 
// 
// Users are encouraged to use these for AFU development
//------------------------------------------------------------------------
// Channel 0 : Memory Reads
typedef struct packed { 
    t_ccip_c0_ReqMemHdr  hdr;            // Request Header 
    logic                valid;          // Request Valid 
} t_if_ccip_c0_Tx; 


// Channel 1 : Memory Writes,  Interrupts, CmpXchg
typedef struct packed { 
    t_ccip_c1_ReqMemHdr  hdr;            // Request Header 
    t_ccip_clData        data;           // Request Data 
    logic                valid;          // Request Wr Valid 
} t_if_ccip_c1_Tx; 

// Channel 2 : MMIO Read response
typedef struct packed { 
    t_ccip_c2_RspMmioHdr    hdr;            // Response Header 
    logic                   mmioRdValid;    // Response Read Valid 
    t_ccip_mmioData         data;           // Response Data 
} t_if_ccip_c2_Tx; 
  
// Wrap all Tx channels
typedef struct packed {
    t_if_ccip_c0_Tx      c0; 
    t_if_ccip_c1_Tx      c1; 
    t_if_ccip_c2_Tx      c2; 
} t_if_ccip_Tx;

// Channel 0: Memory Read response, MMIO Request
typedef struct packed { 
    t_ccip_c0_RspMemHdr  hdr;            //  Rd Response/ MMIO req Header
    t_ccip_clData        data;           //  Rd Data / MMIO req Data 
    // Only one of valid, mmioRdValid and mmioWrValid may be set
    // in a cycle.  When either mmioRdValid or mmioWrValid are true
    // the hdr must be processed specially.  See t_ccip_c0_ReqMmioHdr
    // above.
    logic                rspValid;       //  Rd Response Valid 
    logic                mmioRdValid;    //  MMIO Read Valid
    logic                mmioWrValid;    //  MMIO Write Valid
} t_if_ccip_c0_Rx;

// Channel 1: Memory Writes 
typedef struct packed { 
    t_ccip_c1_RspMemHdr  hdr;            //  Response Header 
    logic                rspValid;       //  Response Valid 
} t_if_ccip_c1_Rx; 

// Wrap all channels 
typedef struct packed { 
    logic                c0TxAlmFull;    //  C0 Request Channel Almost Full 
    logic                c1TxAlmFull;    //  C1 Request Channel Almost Full 

    t_if_ccip_c0_Rx      c0; 
    t_if_ccip_c1_Rx      c1; 
} t_if_ccip_Rx;


typedef union packed { 
    t_ccip_c0_RspMemHdr  rspMemHdr;
    t_ccip_c0_ReqMmioHdr reqMmioHdr;
} t_if_ccip_c0_RxHdr;

endpackage
