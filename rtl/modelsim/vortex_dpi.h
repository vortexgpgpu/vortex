
extern "C" {
    void load_file   (char * filename);
    void dbus_driver(bool clk, unsigned o_m_read_addr, unsigned o_m_evict_addr, bool o_m_valid, svLogicVecVal * o_m_writedata, bool o_m_read_or_write, unsigned cache_banks, unsigned num_words_per_block, svLogicVecVal * i_m_readdata, bool * i_m_ready);
    void ibus_driver(bool clk, unsigned o_m_read_addr, unsigned o_m_evict_addr, bool o_m_valid, svLogicVecVal * o_m_writedata, bool o_m_read_or_write, unsigned cache_banks, unsigned num_words_per_block, svLogicVecVal * i_m_readdata, bool * i_m_ready);
    void io_handler  (bool clk, bool io_valid, unsigned io_data);
    void gracefulExit();
}