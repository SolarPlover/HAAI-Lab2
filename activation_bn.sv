//`timescale 1ns / 1ps



module activation_bn #(parameter NEURON_NB=64, IN_SIZE=64, WIDTH=8)(
    input clk,
    input en,
    input reset,
    input signed[WIDTH-1:0] data_in [0:NEURON_NB-1],
    output reg data_out [0:NEURON_NB-1],
    output layer_done
    );
    integer addr = 0;
    reg done = 0;
    reg signed [4*WIDTH-1:0] out = 0;
    
    //wire signed [4*WIDTH-1:0] temp;

    always @(posedge clk) begin
        if(reset) begin 
            done <= 0;
            addr <= 0;
        end
        else if(en) begin
            if(addr <= NEURON_NB-1) begin
                out <= out+data_in[addr]; //Sum each weighted input
               
            end
            if(addr == NEURON_NB-1) begin //Neuron output available
                done <= 1;
            end else begin
                addr <= addr + 1'b1;
                done <= 0;
            end
        end
    end

    integer i;
    reg compare_done = 0;

    always @(*) begin
        if(done) begin
            for (i = 0; i < NEURON_NB; i = i + 1) begin
                data_out[i] = ((data_in[i] << 6) > out )? 1'b1 : 1'b0;
                if(i == NEURON_NB - 1) begin
                    compare_done = 1;
                end
            // Perform your action here when data_in[i] equals out
            end
        end    
    end
    
   // assign data_out = ( (data_in << 6) > out )? 1'b1 : 1'b0; //Take data_in if > 0, 0 else
    assign layer_done = compare_done;
    
endmodule
