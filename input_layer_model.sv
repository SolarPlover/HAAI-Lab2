//`timescale 1ns / 1ps

module input_layer_model # (parameter NEURON_NB=64, IN_SIZE=784, WIDTH=8)(
    input clk,
    input layer_en,
    input reset,
    input [WIDTH-1:0] in_data [0:IN_SIZE-1],
    input weights [0:NEURON_NB-1][0:IN_SIZE-1],
    //input signed[1:0] biases [0:NEURON_NB-1],
    output signed[4*WIDTH-1:0] neuron_out [0:NEURON_NB-1],
    output layer_done
    );
    
    reg [0:NEURON_NB-1] neuron_done;
    reg done = 1'b0;
    
    neuron_input_layer #(.IN_SIZE(IN_SIZE), .WIDTH(WIDTH)) dense_neuron[0:NEURON_NB-1] (.clk(clk), .en(layer_en), .reset(reset), 
                                                                            .in_data(in_data), .weight(weights), 
                                                                            .neuron_out(neuron_out), .neuron_done(neuron_done)); // Neuron submodules
    always @(posedge clk) begin
        if(neuron_done == '1) begin //All neurons done
            done <= 1'b1;
        end
    end
    
    assign layer_done = done;

    
endmodule