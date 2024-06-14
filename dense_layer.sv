//`timescale 1ns / 1ps


module dense_layer # (parameter NEURON_NB=64, IN_SIZE=64, WIDTH=8)(
    input clk,
    input layer_en,
    input reset,
    input in_data [0:IN_SIZE-1],
    input weights [0:NEURON_NB-1][0:IN_SIZE-1],
    //input signed[1:0] biases [0:NEURON_NB-1],
    output signed[WIDTH-1:0] neuron_out [0:NEURON_NB-1],
    output layer_done
    );
    
    reg [0:NEURON_NB-1] neuron_done;
    reg done = 0;
    
    neuron #(.IN_SIZE(IN_SIZE), .WIDTH(WIDTH)) dense_neuron[0:NEURON_NB-1] (.clk(clk), .en(layer_en), .reset(reset), 
                                                                            .in_data(in_data), .weight(weights), 
                                                                            .neuron_out(neuron_out), .neuron_done(neuron_done)); // Neuron submodules
    always @(posedge clk) begin
        if(neuron_done == '1) begin //All neurons done
            done <= 1;
        end
    end
    
    assign layer_done = done;

    
endmodule
