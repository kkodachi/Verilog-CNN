`ifndef CONV2D_MODULE
`define CONV2D_MODULE

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
    input wire signed [15:0] input_data [0:INPUT_WIDTH-1][0:INPUT_HEIGHT-1][0:INPUT_CHANNELS-1],
    output reg signed [15:0] feature_map [0:INPUT_WIDTH-WINDOW_SIZE][0:INPUT_HEIGHT-WINDOW_SIZE][0:NUM_NEURONS-1],
    output reg conv_done
);

    // Fixed-point multiplication macro
    `define FIXED_MULT(a, b) ((a * b) >>> FIXED_POINT_FRACTIONAL_BITS)

    // Weight initialization and storage
    reg signed [15:0] kernel [0:INPUT_CHANNELS-1][0:WINDOW_SIZE-1][0:WINDOW_SIZE-1][0:NUM_NEURONS-1];
    
    // Convolution state machine
    enum {IDLE, INIT_WEIGHTS, COMPUTING, DONE} state;
    
    // Computation variables
    integer i, j, k, l, m, n;
    reg signed [31:0] conv_accumulator;

    // Weight initialization with pseudo-random generation
    function automatic [15:0] generate_weight();
        reg [31:0] seed;
        begin
            seed = $urandom();
            generate_weight = {seed[15:0]} >>> 1; // Ensure signed, positive weights
        end
    endfunction

    // Convolution process
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            state <= INIT_WEIGHTS;
            conv_done <= 0;
            
            // Initialize weights
            for (k = 0; k < INPUT_CHANNELS; k = k + 1)
                for (l = 0; l < WINDOW_SIZE; l = l + 1)
                    for (m = 0; m < WINDOW_SIZE; m = m + 1)
                        for (n = 0; n < NUM_NEURONS; n = n + 1)
                            kernel[k][l][m][n] <= generate_weight();
        end else begin
            case (state)
                INIT_WEIGHTS: begin
                    if (enable) begin
                        state <= COMPUTING;
                        // Reset feature map
                        for (i = 0; i < INPUT_WIDTH - WINDOW_SIZE + 1; i = i + 1)
                            for (j = 0; j < INPUT_HEIGHT - WINDOW_SIZE + 1; j = j + 1)
                                for (n = 0; n < NUM_NEURONS; n = n + 1)
                                    feature_map[i][j][n] <= 0;
                    end
                end
                
                COMPUTING: begin
                    // Convolution computation with fixed-point arithmetic
                    for (i = 0; i < INPUT_WIDTH - WINDOW_SIZE + 1; i = i + 1) begin
                        for (j = 0; j < INPUT_HEIGHT - WINDOW_SIZE + 1; j = j + 1) begin
                            for (n = 0; n < NUM_NEURONS; n = n + 1) begin
                                conv_accumulator = 0;
                                for (k = 0; k < INPUT_CHANNELS; k = k + 1)
                                    for (l = 0; l < WINDOW_SIZE; l = l + 1)
                                        for (m = 0; m < WINDOW_SIZE; m = m + 1)
                                            conv_accumulator += 
                                                `FIXED_MULT(input_data[i+l][j+m][k], 
                                                            kernel[k][l][m][n]);
                                
                                // Saturate and store result
                                feature_map[i][j][n] <= 
                                    (conv_accumulator > 32767) ? 32767 :
                                    (conv_accumulator < -32768) ? -32768 :
                                    conv_accumulator[15:0];
                            end
                        end
                    end
                    
                    state <= DONE;
                end
                
                DONE: begin
                    conv_done <= 1;
                    state <= IDLE;
                end
                
                default: state <= IDLE;
            endcase
        end
    end
endmodule
`endif