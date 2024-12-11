`timescale 1ns/1ps

module conv2d #(
    parameter INPUT_WIDTH = 64,
    parameter INPUT_HEIGHT = 64,
    parameter INPUT_CHANNELS = 1,
    parameter WINDOW_SIZE = 3,
    parameter NUM_NEURONS = 30,
    parameter FIXED_POINT_FRACTIONAL_BITS = 8
)(
    input wire clk,
    input wire reset,
    input wire enable,
    // Memory interface for input
    input wire signed [15:0] input_data,
    output reg [$clog2(INPUT_WIDTH*INPUT_HEIGHT*INPUT_CHANNELS)-1:0] input_addr,
    input wire input_valid,
    // Memory interface for output
    output reg signed [15:0] feature_map,
    output reg [$clog2((INPUT_WIDTH-WINDOW_SIZE+1)*(INPUT_HEIGHT-WINDOW_SIZE+1)*NUM_NEURONS)-1:0] output_addr,
    output reg output_valid,
    // Control signals
    output reg conv_done
);

    // Local parameters
    localparam INPUT_SIZE = INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS;
    localparam KERNEL_SIZE = INPUT_CHANNELS * WINDOW_SIZE * WINDOW_SIZE * NUM_NEURONS;
    localparam OUTPUT_WIDTH = INPUT_WIDTH - WINDOW_SIZE + 1;
    localparam OUTPUT_HEIGHT = INPUT_HEIGHT - WINDOW_SIZE + 1;
    localparam OUTPUT_SIZE = OUTPUT_WIDTH * OUTPUT_HEIGHT * NUM_NEURONS;

    // Kernel weights memory (distributed RAM)
    reg signed [15:0] kernel_weights [0:KERNEL_SIZE-1];
    reg [$clog2(KERNEL_SIZE)-1:0] kernel_addr;

    // State machine
    reg [2:0] state;
    localparam IDLE = 3'd0;
    localparam INIT_WEIGHTS = 3'd1;
    localparam LOAD_WINDOW = 3'd2;
    localparam COMPUTE = 3'd3;
    localparam STORE_RESULT = 3'd4;
    localparam DONE = 3'd5;

    // Position counters
    reg [$clog2(INPUT_WIDTH)-1:0] i_pos;
    reg [$clog2(INPUT_HEIGHT)-1:0] j_pos;
    reg [$clog2(NUM_NEURONS)-1:0] neuron_cnt;
    reg [$clog2(WINDOW_SIZE)-1:0] win_i, win_j;
    reg [$clog2(INPUT_CHANNELS)-1:0] channel;

    // Computation registers
    reg signed [31:0] conv_sum;
    reg [3:0] window_load_state;
    reg signed [15:0] window_buffer [0:WINDOW_SIZE*WINDOW_SIZE-1];
    
    // Helper functions
    function automatic integer get_input_index;
        input integer x, y, c;
        begin
            get_input_index = x + y * INPUT_WIDTH + c * INPUT_WIDTH * INPUT_HEIGHT;
        end
    endfunction

    function automatic integer get_kernel_index;
        input integer c, x, y, n;
        begin
            get_kernel_index = c * WINDOW_SIZE * WINDOW_SIZE * NUM_NEURONS + 
                             x * WINDOW_SIZE * NUM_NEURONS +
                             y * NUM_NEURONS + n;
        end
    endfunction

    function automatic integer get_output_index;
        input integer x, y, n;
        begin
            get_output_index = x + y * OUTPUT_WIDTH + n * OUTPUT_WIDTH * OUTPUT_HEIGHT;
        end
    endfunction

    // Main state machine
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            state <= INIT_WEIGHTS;
            conv_done <= 0;
            output_valid <= 0;
            i_pos <= 0;
            j_pos <= 0;
            neuron_cnt <= 0;
            win_i <= 0;
            win_j <= 0;
            channel <= 0;
            kernel_addr <= 0;
            conv_sum <= 0;
            window_load_state <= 0;
        end else begin
            case (state)
                INIT_WEIGHTS: begin
                    if (kernel_addr < KERNEL_SIZE) begin
                        kernel_weights[kernel_addr] <= $random;
                        kernel_addr <= kernel_addr + 1;
                    end else begin
                        state <= enable ? LOAD_WINDOW : IDLE;
                        kernel_addr <= 0;
                    end
                end

                LOAD_WINDOW: begin
                    if (window_load_state < WINDOW_SIZE * WINDOW_SIZE) begin
                        input_addr <= get_input_index(
                            i_pos + win_i,
                            j_pos + win_j,
                            channel
                        );
                        if (input_valid) begin
                            window_buffer[window_load_state] <= input_data;
                            if (win_j == WINDOW_SIZE-1) begin
                                win_j <= 0;
                                win_i <= win_i + 1;
                            end else begin
                                win_j <= win_j + 1;
                            end
                            window_load_state <= window_load_state + 1;
                        end
                    end else begin
                        state <= COMPUTE;
                        window_load_state <= 0;
                        win_i <= 0;
                        win_j <= 0;
                    end
                end

                COMPUTE: begin
                    if (win_i < WINDOW_SIZE && win_j < WINDOW_SIZE) begin
                        kernel_addr <= get_kernel_index(
                            channel,
                            win_i,
                            win_j,
                            neuron_cnt
                        );
                        conv_sum <= conv_sum + 
                            ((window_buffer[win_i * WINDOW_SIZE + win_j] * 
                              kernel_weights[kernel_addr]) >>> FIXED_POINT_FRACTIONAL_BITS);
                        
                        if (win_j == WINDOW_SIZE-1) begin
                            win_j <= 0;
                            win_i <= win_i + 1;
                        end else begin
                            win_j <= win_j + 1;
                        end
                    end else begin
                        state <= STORE_RESULT;
                        win_i <= 0;
                        win_j <= 0;
                    end
                end

                STORE_RESULT: begin
                    output_addr <= get_output_index(i_pos, j_pos, neuron_cnt);
                    feature_map <= conv_sum[15:0];
                    output_valid <= 1;
                    conv_sum <= 0;
                    
                    if (neuron_cnt == NUM_NEURONS-1) begin
                        neuron_cnt <= 0;
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
                        neuron_cnt <= neuron_cnt + 1;
                        state <= LOAD_WINDOW;
                    end
                    
                end

                DONE: begin
                    conv_done <= 1;
                    output_valid <= 0;
                    state <= IDLE;
                end

                default: begin // IDLE
                    if (enable) begin
                        state <= LOAD_WINDOW;
                        conv_done <= 0;
                        output_valid <= 0;
                        i_pos <= 0;
                        j_pos <= 0;
                        neuron_cnt <= 0;
                        win_i <= 0;
                        win_j <= 0;
                        channel <= 0;
                        conv_sum <= 0;
                    end
                end
            endcase
        end
    end

endmodule