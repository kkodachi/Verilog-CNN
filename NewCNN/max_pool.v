`timescale 1ns/1ps

module max_pool #(
    parameter INPUT_WIDTH = 64,
    parameter INPUT_HEIGHT = 64,
    parameter INPUT_CHANNELS = 30,
    parameter STRIDE = 2,
    parameter FIXED_POINT_BITS = 8
)(
    input wire clk,
    input wire reset,
    input wire enable,
    // Forward pass
    input wire [15:0] input_data,
    output reg [$clog2(INPUT_WIDTH*INPUT_HEIGHT*INPUT_CHANNELS)-1:0] input_addr,
    input wire input_valid,
    output reg [15:0] pooled_output,
    output reg [$clog2((INPUT_WIDTH/STRIDE)*(INPUT_HEIGHT/STRIDE)*INPUT_CHANNELS)-1:0] output_addr,
    output reg output_valid,
    output reg pool_done,
    // Backpropagation
    input wire [15:0] output_error,
    output reg [15:0] input_error,
    output reg backprop_done
);

    // State machine
    reg [2:0] state;
    localparam IDLE = 3'd0;
    localparam LOAD = 3'd1;
    localparam COMPARE = 3'd2;
    localparam STORE = 3'd3;
    localparam BACKPROP = 3'd4;
    localparam DONE = 3'd5;

    // Window buffer and indices
    reg [15:0] window_buffer [0:STRIDE*STRIDE-1];
    reg [$clog2(STRIDE*STRIDE)-1:0] window_idx;
    reg [$clog2(INPUT_WIDTH/STRIDE)-1:0] out_x;
    reg [$clog2(INPUT_HEIGHT/STRIDE)-1:0] out_y;
    reg [$clog2(INPUT_CHANNELS)-1:0] channel;
    
    // Max value tracking
    reg [15:0] max_val;
    reg [$clog2(STRIDE*STRIDE)-1:0] max_idx;
    
    // Store max positions for backprop
    reg [$clog2(STRIDE*STRIDE)-1:0] max_positions [(INPUT_WIDTH/STRIDE)*(INPUT_HEIGHT/STRIDE)*INPUT_CHANNELS-1:0];

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            state <= IDLE;
            pool_done <= 0;
            backprop_done <= 0;
            output_valid <= 0;
            window_idx <= 0;
            out_x <= 0;
            out_y <= 0;
            channel <= 0;
            max_val <= 16'h8000; // Minimum possible value
        end else begin
            case (state)
                IDLE: begin
                    if (enable) begin
                        state <= LOAD;
                        window_idx <= 0;
                    end
                end

                LOAD: begin
                    if (input_valid) begin
                        window_buffer[window_idx] <= input_data;
                        if (window_idx == 0)
                            max_val <= input_data;
                            
                        if (window_idx == STRIDE*STRIDE-1)
                            state <= COMPARE;
                        else
                            window_idx <= window_idx + 1;
                    end
                end

                COMPARE: begin
                    if (window_buffer[window_idx] > max_val) begin
                        max_val <= window_buffer[window_idx];
                        max_idx <= window_idx;
                    end
                    
                    if (window_idx == STRIDE*STRIDE-1)
                        state <= STORE;
                    else
                        window_idx <= window_idx + 1;
                end

                STORE: begin
                    pooled_output <= max_val;
                    output_valid <= 1;
                    max_positions[output_addr] <= max_idx;
                    
                    if (output_error != 0)
                        state <= BACKPROP;
                    else if (out_x == INPUT_WIDTH/STRIDE-1 && 
                            out_y == INPUT_HEIGHT/STRIDE-1 && 
                            channel == INPUT_CHANNELS-1)
                        state <= DONE;
                    else begin
                        if (out_x == INPUT_WIDTH/STRIDE-1) begin
                            out_x <= 0;
                            if (out_y == INPUT_HEIGHT/STRIDE-1) begin
                                out_y <= 0;
                                channel <= channel + 1;
                            end else
                                out_y <= out_y + 1;
                        end else
                            out_x <= out_x + 1;
                        state <= LOAD;
                    end
                end

                BACKPROP: begin
                    // Propagate error only to max position
                    input_error <= (window_idx == max_positions[output_addr]) ? 
                                 output_error : 0;
                    
                    if (window_idx == STRIDE*STRIDE-1)
                        state <= DONE;
                    else
                        window_idx <= window_idx + 1;
                end

                DONE: begin
                    pool_done <= 1;
                    if (output_error != 0)
                        backprop_done <= 1;
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule