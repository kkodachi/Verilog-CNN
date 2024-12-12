`timescale 1ns/1ps

module conv2d #(
    parameter INPUT_WIDTH = 64,
    parameter INPUT_HEIGHT = 64,
    parameter INPUT_CHANNELS = 1,
    parameter WINDOW_SIZE = 3,
    parameter NUM_NEURONS = 30,
    parameter FIXED_POINT_BITS = 8
)(
    input wire clk,
    input wire reset,
    input wire enable,
    input wire [15:0] input_data,
    output reg [$clog2(INPUT_WIDTH*INPUT_HEIGHT*INPUT_CHANNELS)-1:0] input_addr,
    input wire input_valid,
    output reg [15:0] feature_map,
    output reg [$clog2((INPUT_WIDTH-WINDOW_SIZE+1)*(INPUT_HEIGHT-WINDOW_SIZE+1)*NUM_NEURONS)-1:0] output_addr,
    output reg output_valid,
    output reg conv_done
);

    // State machine
    reg [2:0] state;
    localparam IDLE = 3'd0;
    localparam LOAD = 3'd1;
    localparam COMPUTE = 3'd2;
    localparam STORE = 3'd3;
    localparam DONE = 3'd4;

    // Internal registers
    reg [15:0] window_buffer [0:WINDOW_SIZE*WINDOW_SIZE-1];
    reg [15:0] weight_buffer [0:WINDOW_SIZE*WINDOW_SIZE-1];
    reg [$clog2(WINDOW_SIZE*WINDOW_SIZE)-1:0] window_idx;
    reg [31:0] acc;
    reg [15:0] x_pos, y_pos;

    integer i;

    // Initialize weights (simplified for simulation)
    initial begin
        for (i = 0; i < WINDOW_SIZE*WINDOW_SIZE; i = i + 1)
            weight_buffer[i] = 16'h0100; // 1.0 in fixed point
    end

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            state <= IDLE;
            window_idx <= 0;
            input_addr <= 0;
            output_addr <= 0;
            output_valid <= 0;
            conv_done <= 0;
            x_pos <= 0;
            y_pos <= 0;
            acc <= 0;
        end else begin
            case (state)
                IDLE: begin
                    if (enable && input_valid) begin
                        state <= LOAD;
                        window_idx <= 0;
                        acc <= 0;
                        output_valid <= 0;
                        conv_done <= 0;
                    end
                end

                LOAD: begin
                    if (input_valid) begin
                        window_buffer[window_idx] <= input_data;
                        input_addr <= input_addr + 1;
                        
                        if (window_idx == WINDOW_SIZE*WINDOW_SIZE-1) begin
                            window_idx <= 0;
                            state <= COMPUTE;
                        end else begin
                            window_idx <= window_idx + 1;
                        end
                    end
                end

                COMPUTE: begin
                    // Compute convolution for current window
                    acc <= acc + (window_buffer[window_idx] * weight_buffer[window_idx]);
                    
                    if (window_idx == WINDOW_SIZE*WINDOW_SIZE-1) begin
                        state <= STORE;
                    end else begin
                        window_idx <= window_idx + 1;
                    end
                end

                STORE: begin
                    feature_map <= acc[23:8]; // Fixed point adjustment
                    output_valid <= 1;
                    output_addr <= y_pos * (INPUT_WIDTH-WINDOW_SIZE+1) + x_pos;

                    if (x_pos == INPUT_WIDTH-WINDOW_SIZE) begin
                        x_pos <= 0;
                        if (y_pos == INPUT_HEIGHT-WINDOW_SIZE) begin
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
                    conv_done <= 1;
                    output_valid <= 0;
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule