`ifndef MAX_POOL_MODULE
`define MAX_POOL_MODULE

module max_pool #(
    parameter INPUT_WIDTH = 64,
    parameter INPUT_HEIGHT = 64,
    parameter INPUT_CHANNELS = 30,
    parameter STRIDE = 2
)(
    input wire clk,
    input wire reset,
    input wire enable,
    input wire signed [15:0] feature_map [0:INPUT_WIDTH-1][0:INPUT_HEIGHT-1][0:INPUT_CHANNELS-1],
    output reg signed [15:0] pooled_output [0:INPUT_WIDTH/STRIDE-1][0:INPUT_HEIGHT/STRIDE-1][0:INPUT_CHANNELS-1],
    output reg pool_done
);

    // Pooling state machine
    enum {IDLE, COMPUTING, DONE} state;
    
    // Computation variables
    integer i, j, k, x, y;
    reg signed [15:0] max_value;

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            state <= IDLE;
            pool_done <= 0;
            
            // Reset pooled output
            for (i = 0; i < INPUT_WIDTH/STRIDE; i = i + 1)
                for (j = 0; j < INPUT_HEIGHT/STRIDE; j = j + 1)
                    for (k = 0; k < INPUT_CHANNELS; k = k + 1)
                        pooled_output[i][j][k] <= 0;
        end else begin
            case (state)
                IDLE: begin
                    if (enable) begin
                        state <= COMPUTING;
                        pool_done <= 0;
                    end
                end
                
                COMPUTING: begin
                    // Max pooling computation with error checking
                    for (i = 0; i < INPUT_WIDTH/STRIDE; i = i + 1) begin
                        for (j = 0; j < INPUT_HEIGHT/STRIDE; j = j + 1) begin
                            for (k = 0; k < INPUT_CHANNELS; k = k + 1) begin
                                // Boundary check to prevent out-of-bounds access
                                if (i*STRIDE + STRIDE <= INPUT_WIDTH && 
                                    j*STRIDE + STRIDE <= INPUT_HEIGHT) begin
                                    
                                    max_value = feature_map[i*STRIDE][j*STRIDE][k];
                                    
                                    // Find max in the stride x stride window
                                    for (x = 0; x < STRIDE; x = x + 1)
                                        for (y = 0; y < STRIDE; y = y + 1)
                                            if (feature_map[i*STRIDE+x][j*STRIDE+y][k] > max_value)
                                                max_value = feature_map[i*STRIDE+x][j*STRIDE+y][k];
                                    
                                    pooled_output[i][j][k] <= max_value;
                                end else begin
                                    // Handle boundary cases
                                    pooled_output[i][j][k] <= 0;
                                end
                            end
                        end
                    end
                    
                    state <= DONE;
                end
                
                DONE: begin
                    pool_done <= 1;
                    state <= IDLE;
                end
                
                default: state <= IDLE;
            endcase
        end
    end
endmodule
`endif