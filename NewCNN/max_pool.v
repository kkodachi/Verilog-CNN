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
    // Memory interface for input
    input wire signed [15:0] input_data,
    output reg [$clog2(INPUT_WIDTH*INPUT_HEIGHT*INPUT_CHANNELS)-1:0] input_addr,
    input wire input_valid,
    // Memory interface for output
    output reg signed [15:0] pooled_output,
    output reg [$clog2((INPUT_WIDTH/STRIDE)*(INPUT_HEIGHT/STRIDE)*INPUT_CHANNELS)-1:0] output_addr,
    output reg output_valid,
    // Control signals
    output reg pool_done
);

    // Local parameters
    localparam OUTPUT_WIDTH = INPUT_WIDTH / STRIDE;
    localparam OUTPUT_HEIGHT = INPUT_HEIGHT / STRIDE;
    localparam OUTPUT_SIZE = OUTPUT_WIDTH * OUTPUT_HEIGHT * INPUT_CHANNELS;

    // State machine
    reg [2:0] state;
    localparam IDLE = 3'd0;
    localparam LOAD_WINDOW = 3'd1;
    localparam FIND_MAX = 3'd2;
    localparam STORE_RESULT = 3'd3;
    localparam DONE = 3'd4;

    // Position counters
    reg [$clog2(INPUT_WIDTH)-1:0] i_pos;
    reg [$clog2(INPUT_HEIGHT)-1:0] j_pos;
    reg [$clog2(INPUT_CHANNELS)-1:0] channel;
    reg [$clog2(STRIDE)-1:0] win_i, win_j;

    // Computation registers
    reg signed [15:0] max_value;
    reg [3:0] window_load_state;
    reg signed [15:0] window_buffer [0:STRIDE*STRIDE-1];

    // Helper functions
    function automatic integer get_input_index;
        input integer x, y, c;
        begin
            get_input_index = x + y * INPUT_WIDTH + c * INPUT_WIDTH * INPUT_HEIGHT;
        end
    endfunction

    function automatic integer get_output_index;
        input integer x, y, c;
        begin
            get_output_index = x + y * OUTPUT_WIDTH + c * OUTPUT_WIDTH * OUTPUT_HEIGHT;
        end
    endfunction

    // Main state machine
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            state <= IDLE;
            pool_done <= 0;
            output_valid <= 0;
            i_pos <= 0;
            j_pos <= 0;
            channel <= 0;
            win_i <= 0;
            win_j <= 0;
            max_value <= 16'h8000; // Minimum possible value
            window_load_state <= 0;
        end else begin
            case (state)
                IDLE: begin
                    if (enable) begin
                        state <= LOAD_WINDOW;
                        pool_done <= 0;
                        output_valid <= 0;
                    end
                end

                LOAD_WINDOW: begin
                    if (window_load_state < STRIDE * STRIDE) begin
                        input_addr <= get_input_index(
                            i_pos * STRIDE + win_i,
                            j_pos * STRIDE + win_j,
                            channel
                        );
                        if (input_valid) begin
                            window_buffer[window_load_state] <= input_data;
                            if (win_j == STRIDE-1) begin
                                win_j <= 0;
                                win_i <= win_i + 1;
                            end else begin
                                win_j <= win_j + 1;
                            end
                            window_load_state <= window_load_state + 1;
                        end
                    end else begin
                        state <= FIND_MAX;
                        window_load_state <= 0;
                        win_i <= 0;
                        win_j <= 0;
                    end
                end

                FIND_MAX: begin
                    if (win_i < STRIDE && win_j < STRIDE) begin
                        if (window_buffer[win_i * STRIDE + win_j] > max_value) begin
                            max_value <= window_buffer[win_i * STRIDE + win_j];
                        end
                        if (win_j == STRIDE-1) begin
                            win_j <= 0;
                            win_i <= win_i + 1;
                        end else begin
                            win_j <= win_j + 1;
                        end
                        if (win_i == STRIDE-1 && win_j == STRIDE-1) begin
                            state <= STORE_RESULT;
                        end
                    end
                end

                STORE_RESULT: begin
                    output_addr <= get_output_index(i_pos, j_pos, channel);
                    pooled_output <= max_value;
                    output_valid <= 1;
                    win_i <= 0;
                    win_j <= 0;
                    max_value <= 16'h8000; // Reset max value

                    if (channel == INPUT_CHANNELS-1) begin
                        channel <= 0;
                        if (j_pos == OUTPUT_HEIGHT-1) begin
                            j_pos <= 0;
                            if (i_pos == OUTPUT_WIDTH-1) begin
                                state <= DONE;
                            end else begin
                                i_pos <= i_pos + 1;
                                state <= LOAD_WINDOW;
                            end
                        end else begin
                            j_pos <= j_pos + 1;
                            state <= LOAD_WINDOW;
                        end
                    end else begin
                        channel <= channel + 1;
                        state <= LOAD_WINDOW;
                    end
                end

                DONE: begin
                    pool_done <= 1;
                    output_valid <= 0;
                    state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule