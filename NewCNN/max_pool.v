`timescale 1ns/1ps

module max_pool #(
    parameter INPUT_WIDTH = 64,
    parameter INPUT_HEIGHT = 64,
    parameter INPUT_CHANNELS = 30,
    parameter STRIDE = 2
)(
    input wire clk,
    input wire reset,
    input wire enable,
    input wire [15:0] input_data,
    input wire input_valid,
    output reg [15:0] pooled_output,
    output reg output_valid,
    output reg pool_done
);

    // State machine
    reg [2:0] state;
    localparam IDLE = 3'd0;
    localparam LOAD = 3'd1;
    localparam COMPARE = 3'd2;
    localparam STORE = 3'd3;
    localparam DONE = 3'd4;

    // Internal registers
    reg [15:0] window [0:STRIDE*STRIDE-1];
    reg [15:0] max_val;
    reg [$clog2(STRIDE*STRIDE)-1:0] window_idx;
    reg [15:0] x_pos, y_pos;

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            state <= IDLE;
            window_idx <= 0;
            output_valid <= 0;
            pool_done <= 0;
            x_pos <= 0;
            y_pos <= 0;
            max_val <= 16'h8000; // Minimum value
        end else begin
            case (state)
                IDLE: begin
                    if (enable && input_valid) begin
                        state <= LOAD;
                        window_idx <= 0;
                        max_val <= 16'h8000;
                        output_valid <= 0;
                        pool_done <= 0;
                    end
                end

                LOAD: begin
                    if (input_valid) begin
                        window[window_idx] <= input_data;
                        
                        if (window_idx == STRIDE*STRIDE-1) begin
                            window_idx <= 0;
                            state <= COMPARE;
                        end else begin
                            window_idx <= window_idx + 1;
                        end
                    end
                end

                COMPARE: begin
                    if (window[window_idx] > max_val) begin
                        max_val <= window[window_idx];
                    end
                    
                    if (window_idx == STRIDE*STRIDE-1) begin
                        state <= STORE;
                    end else begin
                        window_idx <= window_idx + 1;
                    end
                end

                STORE: begin
                    pooled_output <= max_val;
                    output_valid <= 1;

                    if (x_pos == (INPUT_WIDTH/STRIDE)-1) begin
                        x_pos <= 0;
                        if (y_pos == (INPUT_HEIGHT/STRIDE)-1) begin
                            state <= DONE;
                        end else begin
                            y_pos <= y_pos + 1;
                            state <= LOAD;
                        end
                    end else begin
                        x_pos <= x_pos + 1;
                        state <= LOAD;
                    end
                end

                DONE: begin
                    pool_done <= 1;
                    output_valid <= 0;
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule