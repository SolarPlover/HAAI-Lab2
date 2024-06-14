//`timescale 1ns / 1ps


module neural_network(
    input clk,
    input enable,
    input reset,
    input [7:0] img [0:783],
    output signed [7:0] digit_out [0:9],
    output NN_done
    );

    /* input layer */
    reg input_enable = 0;
    reg test = 0;
    reg input_out[0:63];
    wire finished_input;
    input_layer INPUT_layer (.clk(clk), .enable(input_enable), .reset(reset), .img_data(img), .data_out(input_out), .layer_done(finished_input));

    always @(posedge clk) begin
        if(reset) begin
            input_enable <= 0;
        end
        else if(enable) begin
            if(finished_input) begin 
                input_enable <= 0;
            end
            else input_enable <= 1; 
        end
    end


    /*hidden later */
    reg hidden_enable;
    reg hidden_out [0:63];
    wire finished_hidden;
    hidden_layer HIDDEN_layer (.clk(clk), .enable(hidden_enable), .reset(reset), .in_data(input_out), .layer_out(hidden_out), .layer_done(finished_hidden));

    always @(posedge clk) begin
        if(reset) begin
            hidden_enable <= 0;
        end
        else if(enable) begin
            if(finished_input == 1 && finished_hidden == 0) begin 
                hidden_enable <= 1;
            end
            else hidden_enable <= 0; 
        end
    end

    /*output later */
    reg output_enable;
    reg signed [7:0] output_out [0:9];
    wire finished_out;
    output_layer OUT_layer (.clk(clk), .enable(output_enable), .reset(reset), .in_data(hidden_out), .digit_out(output_out), .layer_done(finished_out));

    always @(posedge clk) begin
        if(reset) begin
            output_enable <= 0;
        end
        else if(enable) begin
            if(finished_hidden == 1 && finished_out == 0) begin 
                output_enable <= 1;
            end
            else output_enable <= 0; 
            if(finished_out == 1)begin
                test = 1;
            end
        end
    end


    assign NN_done = finished_out;
    assign digit_out = output_out;
    
    
    
endmodule
