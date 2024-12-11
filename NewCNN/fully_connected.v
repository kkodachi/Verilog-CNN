`timescale 1ns/1ps

module fully_connected #(
    parameter INPUT_SIZE = 120,
    parameter OUTPUT_SIZE = 10,
    parameter FIXED_POINT_FRACTIONAL_BITS = 8
)(
    input wire clk,
    input wire reset,
    input wire enable,
    // Memory interface for input
    input wire signed [15:0] input_data,
    output reg [$clog2(INPUT_SIZE)-1:0] input_addr,
    input wire input_valid,
    // Memory interface for output
    output reg signed [15:0] output_data,
    output reg [$clog2(OUTPUT_SIZE)-1:0] output_addr,
    output reg output_valid,
    // Control signals
    output reg fc_done
);

    // Fixed-point multiplication macro
    `define FIXED_MULT(a, b) ((a * b) >>> FIXED_POINT_FRACTIONAL_BITS)

    // Weight and bias memory
    reg signed [15:0] weights [0:INPUT_SIZE*OUTPUT_SIZE-1];
    reg signed [15:0] biases [0:OUTPUT_SIZE-1];
    
    // State machine
    reg [2:0] state;
    parameter IDLE = 3'd0;
    parameter INIT_WEIGHTS = 3'd1;
    parameter LOAD_INPUT = 3'd2;
    parameter COMPUTE = 3'd3;
    parameter STORE_RESULT = 3'd4;
    parameter DONE = 3'd5;

    // Counters and computation registers
    reg [$clog2(INPUT_SIZE)-1:0] input_cnt;
    reg [$clog2(OUTPUT_SIZE)-1:0] output_cnt;
    reg [$clog2(INPUT_SIZE*OUTPUT_SIZE)-1:0] weight_addr;
    reg signed [31:0] accumulator;
    reg signed [15:0] input_buffer;

    // Weight initialization counter
    reg [$clog2(INPUT_SIZE*OUTPUT_SIZE):0] init_counter;
    reg bias_init;

    // Helper function for weight index calculation
    function automatic integer get_weight_index;
        input integer in_idx, out_idx;
        begin
            get_weight_index = in_idx + out_idx * INPUT_SIZE;
        end
    endfunction

    // Main state machine
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            state <= INIT_WEIGHTS;
            fc_done <= 0;
            output_valid <= 0;
            input_cnt <= 0;
            output_cnt <= 0;
            weight_addr <= 0;
            accumulator <= 0;
            input_addr <= 0;
            output_addr <= 0;
            init_counter <= 0;
            bias_init <= 0;
        end else begin
            case (state)
                INIT_WEIGHTS: begin
                    if (!bias_init) begin
                        if (init_counter < OUTPUT_SIZE) begin
                            biases[init_counter] <= $random;
                            init_counter <= init_counter + 1;
                        end else begin
                            bias_init <= 1;
                            init_counter <= 0;
                        end
                    end else begin
                        if (init_counter < INPUT_SIZE * OUTPUT_SIZE) begin
                            weights[init_counter] <= $random;
                            init_counter <= init_counter + 1;
                        end else begin
                            state <= IDLE;
                            init_counter <= 0;
                        end
                    end
                end

                IDLE: begin
                    if (enable) begin
                        state <= LOAD_INPUT;
                        output_valid <= 0;
                    end
                end

                LOAD_INPUT: begin
                    input_addr <= input_cnt;
                    if (input_valid) begin
                        input_buffer <= input_data;
                        state <= COMPUTE;
                    end
                end

                COMPUTE: begin
                    weight_addr <= get_weight_index(input_cnt, output_cnt);
                    
                    if (input_cnt == 0) begin
                        accumulator <= {biases[output_cnt], {FIXED_POINT_FRACTIONAL_BITS{1'b0}}};
                    end
                    
                    accumulator <= accumulator + 
                        ((input_buffer * weights[weight_addr]) >>> FIXED_POINT_FRACTIONAL_BITS);
                    
                    if (input_cnt == INPUT_SIZE - 1) begin
                        input_cnt <= 0;
                        state <= STORE_RESULT;
                    end else begin
                        input_cnt <= input_cnt + 1;
                        state <= LOAD_INPUT;
                    end
                end

                STORE_RESULT: begin
                    output_data <= (accumulator > 32767) ? 16'h7FFF :
                                 (accumulator < -32768) ? 16'h8000 :
                                 accumulator[15:0];
                    output_addr <= output_cnt;
                    output_valid <= 1;
                    accumulator <= 0;

                    if (output_cnt == OUTPUT_SIZE - 1) begin
                        output_cnt <= 0;
                        state <= DONE;
                    end else begin
                        output_cnt <= output_cnt + 1;
                        state <= LOAD_INPUT;
                    end
                end

                DONE: begin
                    fc_done <= 1;
                    output_valid <= 0;
                    state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule