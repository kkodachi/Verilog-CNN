module ModelTraining #(
    parameter IMG_HEIGHT = 64,
    parameter IMG_WIDTH = 64,
    parameter KERNEL = 3,
    parameter CHANNELS = 1,
    parameter NEURONS = 30,
    parameter FC_INPUT_SIZE = 120,
    parameter FC_OUTPUT_SIZE = 10
)(
    input clk,
    input rst,
    input start,
    input [7:0] img [0:IMG_WIDTH*IMG_HEIGHT-1], // flattened for synthesis
    input [7:0] kernel [0:KERNEL*KERNEL-1], // flattened for synthesis
    input [FC_OUTPUT_SIZE-1:0] ground_truth, // one-hot encoded
    input [15:0] weights [0:FC_INPUT_SIZE*FC_OUTPUT_SIZE-1], // flattened for synthesis
    input [15:0] bias [0:FC_OUTPUT_SIZE-1],
    output reg done,
    output reg [31:0] final_output [0:FC_OUTPUT_SIZE-1],
    output reg [31:0] loss
);

    // State declarations for control flow
    reg [3:0] state;
    localparam  IDLE = 4'b0000,
                CONV2D_F = 4'b0001,      // Forward Convolution
                MAXPOOL_F = 4'b0010,     // Forward Max-Pooling
                FC_F = 4'b0011,          // Forward Fully Connected
                CALC_LOSS = 4'b0100,     // Loss Calculation
                DONE = 4'b1001;

    // Control signals for starting each module
    reg start_conv2d_f, start_maxpool_f, start_fc_f, start_loss_calc;
    
    // Done signals from each module
    wire conv2d_f_done, maxpool_f_done, fc_f_done, loss_done;

    // Intermediate signals for data flow between modules
    wire [15:0] featureMap [0:(IMG_WIDTH-KERNEL)*(IMG_HEIGHT-KERNEL)-1];
    wire [15:0] pooled_output [0:((IMG_WIDTH-KERNEL+1)/2)*((IMG_HEIGHT-KERNEL+1)/2)-1];
    reg [15:0] flattened_output [0:FC_INPUT_SIZE-1];

    // Instantiate Conv2d Forward
    Conv2d_Forward #(
        .IMG_HEIGHT(IMG_HEIGHT),
        .IMG_WIDTH(IMG_WIDTH),
        .KERNEL(KERNEL),
        .CHANNELS(CHANNELS),
        .NEURONS(NEURONS)
    ) conv2d_f (
        .clk(clk),
        .rst(rst),
        .start(start_conv2d_f),
        .img(img),
        .kernel(kernel),
        .featureMap(featureMap),
        .done(conv2d_f_done)
    );

    // Instantiate MaxPool Forward
    MaxPool_Forward #(
        .FM_height(IMG_HEIGHT-KERNEL+1),
        .FM_width(IMG_WIDTH-KERNEL+1),
        .KERNEL(2),
        .STRIDE(2)
    ) maxpool_f (
        .clk(clk),
        .rst(rst),
        .start(start_maxpool_f),
        .featureMap(featureMap),
        .pooled_output(pooled_output),
        .done(maxpool_f_done)
    );

    // Instantiate Fully Connected Forward
    FullyConnect_Forward #(
        .input_size(FC_INPUT_SIZE),
        .output_size(FC_OUTPUT_SIZE)
    ) fc (
        .clk(clk),
        .rst(rst),
        .start(start_fc_f),
        .input_data(flattened_output),
        .weights(weights),
        .bias(bias),
        .output_data(final_output),
        .done(fc_f_done)
    );

    // Instantiate Loss Calculation
    L1Loss #(
        .FC_OUTPUT_SIZE(FC_OUTPUT_SIZE)
    ) l1_loss (
        .clk(clk),
        .rst(rst),
        .start(start_loss_calc),
        .predicted_probs(final_output),
        .ground_truth(ground_truth),
        .loss(loss),
        .done(loss_done)
    );

    // Sequential control of state transitions
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            // Reset all state and control signals
            state <= IDLE;
            done <= 0;
            start_conv2d_f <= 0;
            start_maxpool_f <= 0;
            start_fc_f <= 0;
            start_loss_calc <= 0;
            $display("ModelTraining: Resetting...");
        end else begin
            case (state)
                IDLE: begin
                    if (start) begin
                        $display("ModelTraining: Starting training process...");
                        start_conv2d_f <= 1;
                        state <= CONV2D_F;
                    end
                end

                // Convolution Forward Pass
                CONV2D_F: begin
                    start_conv2d_f <= 0; // De-assert start after launching
                    if (conv2d_f_done) begin
                        $display("ModelTraining: Convolution forward pass done");
                        start_maxpool_f <= 1;
                        state <= MAXPOOL_F;
                    end
                end

                // Max Pooling Forward Pass
                MAXPOOL_F: begin
                    start_maxpool_f <= 0; // De-assert start
                    if (maxpool_f_done) begin
                        $display("ModelTraining: Max pooling forward pass done");
                        // Flatten pooled output for fully connected layer
                        integer idx = 0;
                        integer i, j;
                        for (i = 0; i < (IMG_WIDTH-KERNEL+1)/2; i = i + 1) begin
                            for (j = 0; j < (IMG_HEIGHT-KERNEL+1)/2; j = j + 1) begin
                                flattened_output[idx] = pooled_output[i*(IMG_WIDTH-KERNEL+1)/2 + j];
                                idx = idx + 1;
                            end
                        end
                        start_fc_f <= 1;
                        state <= FC_F;
                    end
                end

                // Fully Connected Forward Pass
                FC_F: begin
                    start_fc_f <= 0;
                    if (fc_f_done) begin
                        $display("ModelTraining: Fully connected forward pass done");
                        start_loss_calc <= 1;
                        state <= CALC_LOSS;
                    end
                end

                // Calculate Loss
                CALC_LOSS: begin
                    start_loss_calc <= 0;
                    if (loss_done) begin
                        $display("ModelTraining: Loss calculation complete. Loss: %d", loss);
                        state <= DONE;
                    end
                end

                // End of training process
                DONE: begin
                    $display("ModelTraining: Training process completed.");
                    done <= 1;
                    state <= IDLE;
                end
            endcase
        end
    end
endmodule
