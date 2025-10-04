import ai.onnxruntime.*;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.*;

public class ONNXSojaPredict {

    private OrtEnvironment env;
    private OrtSession session;
    private List<String> classes;

    public ONNXSojaPredict() throws OrtException, IOException {
        env = OrtEnvironment.getEnvironment();

        // carrega modelo do resources como byte[]
        try (InputStream modelStream = getClass().getResourceAsStream("/resnet50_soja.onnx")) {
            if (modelStream == null) {
                throw new RuntimeException("Modelo ONNX não encontrado no JAR!");
            }
            byte[] modelBytes = modelStream.readAllBytes();
            OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
            session = env.createSession(modelBytes, opts);
        }

        // tenta ler metadados
        Map<String, String> metadata = session.getMetadata().getCustomMetadata();
        if (metadata.containsKey("classes")) {
            classes = Arrays.asList(metadata.get("classes").split(","));
        } else {
            classes = new ArrayList<>();
        }
    }

    public String predict(float[] inputData) throws OrtException {
        long[] shape = {1, 3, 224, 224};
        try (OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputData), shape)) {
            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put(session.getInputNames().iterator().next(), inputTensor);

            try (OrtSession.Result results = session.run(inputs)) {
                float[] probs = ((float[][]) results.get(0).getValue())[0];

                // softmax
                float max = Float.NEGATIVE_INFINITY;
                for (float v : probs) if (v > max) max = v;

                float sum = 0f;
                for (int i = 0; i < probs.length; i++) {
                    probs[i] = (float) Math.exp(probs[i] - max);
                    sum += probs[i];
                }
                for (int i = 0; i < probs.length; i++) probs[i] /= sum;

                // top1
                int maxIdx = 0;
                for (int i = 1; i < probs.length; i++)
                    if (probs[i] > probs[maxIdx]) maxIdx = i;

                String label = classes.isEmpty() ? "Classe_" + maxIdx : classes.get(maxIdx);
                return label + " (" + String.format("%.2f", probs[maxIdx] * 100) + "%)";
            }
        }
    }

    public void close() throws OrtException {
        session.close();
        env.close();
    }
}
