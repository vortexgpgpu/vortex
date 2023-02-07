onerror {resume}
quietly WaveActivateNextPane {} 0
add wave -noupdate /tb_ahb/ahbif/ADDR_WIDTH
add wave -noupdate /tb_ahb/ahbif/DATA_WIDTH
add wave -noupdate /tb_ahb/ahbif/HRESETn
add wave -noupdate /tb_ahb/ahbif/HREADY
add wave -noupdate /tb_ahb/ahbif/HMASTLOCK
add wave -noupdate /tb_ahb/ahb_mod/state
add wave -noupdate /tb_ahb/ahb_mod/next_state
add wave -noupdate -color Cyan /tb_ahb/ahb_mod/range_error
add wave -noupdate -color Gold /tb_ahb/ahbif/HCLK
add wave -noupdate -color {Spring Green} /tb_ahb/ahbif/HSEL
add wave -noupdate -color {Medium Violet Red} /tb_ahb/ahbif/HWRITE
add wave -noupdate -color Cyan /tb_ahb/ahbif/HREADYOUT
add wave -noupdate -color Aquamarine /tb_ahb/ahbif/HRESP
add wave -noupdate -color Gold -radix hexadecimal /tb_ahb/ahbif/HADDR
add wave -noupdate -color {Violet Red} /tb_ahb/bpif/request_stall
add wave -noupdate /tb_ahb/ahbif/HTRANS
add wave -noupdate /tb_ahb/ahbif/HBURST
add wave -noupdate /tb_ahb/ahbif/HSIZE
add wave -noupdate /tb_ahb/ahbif/HWDATA
add wave -noupdate /tb_ahb/ahbif/HRDATA
add wave -noupdate /tb_ahb/ahbif/HWSTRB
add wave -noupdate -group {BFM SIGS} /tb_ahb/periphBFM/dataIndexWidth
add wave -noupdate -group {BFM SIGS} /tb_ahb/periphBFM/CLK
add wave -noupdate -group {BFM SIGS} /tb_ahb/periphBFM/nRST
add wave -noupdate -group {BFM SIGS} /tb_ahb/periphBFM/data
add wave -noupdate -group {BFM SIGS} -expand /tb_ahb/periphBFM/ndata
add wave -noupdate -group {BFM SIGS} /tb_ahb/periphBFM/waited
add wave -noupdate -group {BFM SIGS} /tb_ahb/periphBFM/nwaited
add wave -noupdate -group {BFM SIGS} /tb_ahb/periphBFM/errorOccured
add wave -noupdate -expand -group {BPIF SIGS} /tb_ahb/bpif/ADDR_WIDTH
add wave -noupdate -expand -group {BPIF SIGS} /tb_ahb/bpif/DATA_WIDTH
add wave -noupdate -expand -group {BPIF SIGS} /tb_ahb/bpif/wen
add wave -noupdate -expand -group {BPIF SIGS} /tb_ahb/bpif/ren
add wave -noupdate -expand -group {BPIF SIGS} /tb_ahb/bpif/addr
add wave -noupdate -expand -group {BPIF SIGS} /tb_ahb/bpif/error
add wave -noupdate -expand -group {BPIF SIGS} /tb_ahb/bpif/strobe
add wave -noupdate -expand -group {BPIF SIGS} /tb_ahb/bpif/wdata
add wave -noupdate -expand -group {BPIF SIGS} /tb_ahb/bpif/rdata
add wave -noupdate -expand -group {BPIF SIGS} /tb_ahb/bpif/is_burst
add wave -noupdate -expand -group {BPIF SIGS} /tb_ahb/bpif/burst_type
add wave -noupdate -expand -group {BPIF SIGS} /tb_ahb/bpif/burst_length
add wave -noupdate -expand -group {BPIF SIGS} /tb_ahb/bpif/secure_transfer
TreeUpdate [SetDefaultTree]
WaveRestoreCursors {{Cursor 1} {2888642732 ps} 0}
quietly wave cursor active 1
configure wave -namecolwidth 150
configure wave -valuecolwidth 100
configure wave -justifyvalue left
configure wave -signalnamewidth 1
configure wave -snapdistance 10
configure wave -datasetprefix 0
configure wave -rowmargin 4
configure wave -childrowmargin 2
configure wave -gridoffset 0
configure wave -gridperiod 1
configure wave -griddelta 40
configure wave -timeline 0
configure wave -timelineunits ns
update
WaveRestoreZoom {2888640058 ps} {2888778426 ps}
