module FullyConnect_Forward #(
    parameter input_size = 120,
    parameter output_size = 10
)(
    input clk,
    input rst,
    input [15:0] input_data [0:input_size-1],
    input [15:0] weights [0:input_size-1][0:output_size-1], // matrix of weights
    input [15:0] bias [0:output_size-1],
    output reg [31:0] output_data [0:output_size-1], // fully connected output
    output reg done
);
    integer i, j;
    reg [31:0] sum;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            done <= 0;
        end else begin
            for (i = 0; i < output_size; i = i + 1) begin
                sum = 0;
                for (j = 0; j < input_size; j = j + 1) begin
                    sum = sum + input_data[j] * weights[j][i];
                end
                output_data[i] <= sum + bias[i];
            end
            done <= 1;
        end
    end
endmodule

module FullyConnect_Backward #(
    parameter input_size = 120,
    parameter output_size = 10
)(
    input clk,
    input rst,
    input [15:0] input_data [0:input_size-1],
    input [31:0] output_data [0:output_size-1], // fully connected output
    input [15:0] lossGrad_output [0:output_size-1],
    input [15:0] weights [0:input_size-1][0:output_size-1], // matrix of weights
    output reg [15:0] lossGrad_weights [0:output_size-1][0:input_size-1],
    output reg [15:0] lossGrad_bias [0:output_size-1],
    output reg [15:0] lossGrad_input [0:input_size-1], // Gradients for input
    output reg done
);
    integer i, j;
    reg [31:0] sum;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            done <= 0;
        end else begin
            for (i = 0; i < output_size; i = i + 1) begin
                lossGrad_bias[i] <= lossGrad_output[i];
                for (j = 0; j < input_size; j = j + 1) begin
                    lossGrad_weights[j][i] <= lossGrad_output[i] * input_data[j];
                end
            end

            for (j = 0; j < input_size; j = j + 1) begin
                sum = 0;
                for (i = 0; i < output_size; i = i + 1) begin
                    sum = sum + lossGrad_output[i] * weights[j][i];
                end
                lossGrad_input[j] <= sum;
            end
            done <= 1;
        end
    end
endmodule
