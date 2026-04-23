#ifndef REMOTE_BITBANG_H
#define REMOTE_BITBANG_H

#include <stdint.h>

#include "jtag_dtm.h"

class remote_bitbang_t
{
public:
  // Constructor: Creates a TCP server listening on the specified port for OpenOCD remote bitbang connections.
  // Use case: Initializes the server that will receive JTAG commands from OpenOCD and forward them to the DTM.
  remote_bitbang_t(uint16_t port, jtag_dtm_t *tap);

  // Processes one iteration of the server loop: accepts connections or executes pending JTAG commands.
  // Use case: Called repeatedly in the main loop to handle incoming OpenOCD connections and JTAG protocol.
  void tick();

private:
  jtag_dtm_t *tap;

  int socket_fd;
  int client_fd;

  static const ssize_t buf_size = 64 * 1024;
  char send_buf[buf_size];
  char recv_buf[buf_size];
  ssize_t recv_start, recv_end;

  // Accepts a new client connection if one is waiting (non-blocking).
  // Use case: Called when no client is connected to check for and accept new OpenOCD connections.
  void accept();
  
  // Executes remote bitbang protocol commands from the connected client.
  // Use case: Processes JTAG pin control commands ('0'-'7'), reads TDO ('R'), and handles protocol flow.
  void execute_commands();
};

#endif
