# Function to export netlist to a Graphviz DOT file
proc export_netlist {dot_file_name} {
  # Open the DOT file for writing
  set dot_file [open $dot_file_name "w"]

  # Start the DOT graph definition
  puts $dot_file "digraph Netlist {"
  puts $dot_file "rankdir=LR;"  ;# Set the graph direction from left to right

  # Extract and add cells to the graph
  foreach cell [get_cells -hierarchical] {
    set cell_name [get_property NAME $cell]
    set cell_type [get_property REF_NAME $cell]
    puts $dot_file "\"$cell_name\" \[label=\"$cell_name\\n($cell_type)\", shape=box\];"
  }

  # Extract and add ports to the graph
  foreach port [get_ports] {
    set port_name [get_property NAME $port]
    set direction [get_property DIRECTION $port]
    set shape "ellipse"

    # Color code input and output ports for easier identification
    if {$direction == "IN"} {
      set color "lightblue"
    } else {
      set color "lightgreen"
    }
    puts $dot_file "\"$port_name\" \[label=\"$port_name\", shape=$shape, style=filled, fillcolor=$color\];"
  }

  # Traverse nets and create edges between ports and pins
  foreach net [get_nets -hierarchical] {
    set net_name [get_property NAME $net]

    # Find source and destination pins
    set source_pin ""
    set sink_pins {}

    foreach pin [get_pins -of_objects $net] {
      set direction [get_property DIRECTION $pin]
      set cell [get_cells -of_objects $pin]
      set pin_name [get_property NAME $pin]

      if {$direction == "OUT"} {
        # Set as source pin
        set source_pin "$cell/$pin_name"
      } else {
        # Collect as sink pin
        lappend sink_pins "$cell/$pin_name"
      }
    }

    # Output edges from source to all sinks
    if {$source_pin != ""} {
      foreach sink_pin $sink_pins {
        puts $dot_file "\"$source_pin\" -> \"$sink_pin\" \[label=\"$net_name\"\];"
      }
    }
  }

  # End the DOT graph definition
  puts $dot_file "}"

  # Close the DOT file
  close $dot_file
  puts "Netlist exported to DOT file: $dot_file_name"
}

# Run the export function
export_netlist "netlist.dot"