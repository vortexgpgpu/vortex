```mermaid
flowchart TB
  subgraph CL["Cluster Level (VX_cache_cluster.sv)"]
    CI[Core Inputs NUM_INPUTS]
    CA["Core Arbiter VX_mem_arb (R)"]
    subgraph CACHES["Cache Units (NUM_CACHES)"]
      CTOP1[VX_cache_top.sv]
      CWRAP1[VX_cache_wrap.sv]
      BYPASS[VX_cache_bypass.sv]
      subgraph TLB["TLB Path"]
        TWRAP[VX_tlb_wrap.sv]
        TMAIN[VX_tlb.sv]
        subgraph TBANKS["TLB Banks (NUM_BANKS)"]
          TLB0[VX_tlb_bank.sv]
          TLB1[VX_tlb_bank.sv]
          TLBn[...]
        end
        PTW[VX_ptw.sv]
      end
      subgraph CORECACHE["Cache Core"]
        CACHE1[VX_cache.sv]
        RXBAR["Core Req Xbar VX_stream_xbar (R)"]
        subgraph BANKS["Cache Banks (NUM_BANKS)"]
          B0[Bank 0 VX_cache_bank.sv]
          B1[Bank 1 VX_cache_bank.sv]
          Bn[...]
        end
        MRARB["Mem Req Arb VX_stream_arb (R)"]
        MROUTER["Mem Rsp Xbar VX_stream_omega (R)"]
        RSPXBAR["Core Rsp Xbar VX_stream_xbar (R)"]
      end
    end
    MA["Mem Arbiter VX_mem_arb (R)"]
    MEM[Memory System]
  end

  CI --> CA --> CTOP1 --> CWRAP1
  CWRAP1 -- VM_ENABLED --> TWRAP
  TWRAP --> TMAIN --> TLB0
  TMAIN -. miss .-> PTW
  TWRAP --> CACHE1

  CACHE1 --> RXBAR --> B0
  B0 --> MRARB --> MA --> MEM
  MEM --> MROUTER --> B0
  B0 --> RSPXBAR --> CWRAP1

  CWRAP1 --> BYPASS --> MA