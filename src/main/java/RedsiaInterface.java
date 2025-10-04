// RedsiaInterface.java
import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.io.File;

public class RedsiaInterface extends JFrame {

    private JLabel labelImage, labelResult;
    private JButton btnSelect;
    private ONNXSojaPredict predictor;

    public RedsiaInterface() throws Exception {
        super("Classificador de Soja");
        setLayout(new BorderLayout());
        setSize(500, 500);
        setDefaultCloseOperation(EXIT_ON_CLOSE);

        // Inicializa ONNX predictor
        predictor = new ONNXSojaPredict();


        // Labels
        labelImage = new JLabel("Nenhuma imagem selecionada", SwingConstants.CENTER);
        labelResult = new JLabel("Resultado: ", SwingConstants.CENTER);

        // Botão para selecionar arquivo
        btnSelect = new JButton("Selecionar Imagem");
        btnSelect.addActionListener(e -> selectImage());

        add(labelImage, BorderLayout.CENTER);
        add(labelResult, BorderLayout.SOUTH);
        add(btnSelect, BorderLayout.NORTH);

        setVisible(true);
    }

    private void selectImage() {
        JFileChooser chooser = new JFileChooser();
        int res = chooser.showOpenDialog(this);
        if (res == JFileChooser.APPROVE_OPTION) {
            File file = chooser.getSelectedFile();
            labelImage.setIcon(new ImageIcon(file.getAbsolutePath()));
            labelImage.setText("");

            try {
                float[] data = RedsiaUtils.preprocessImage(file);
                String result = predictor.predict(data);
                labelResult.setText("Resultado: " + result);
            } catch (Exception ex) {
                labelResult.setText("Erro: " + ex.getMessage());
            }
        }
    }

    public static void main(String[] args) throws Exception {
        SwingUtilities.invokeLater(() -> {
            try {
                new RedsiaInterface();
            } catch (Exception e) {
                e.printStackTrace();
            }
        });
    }
}

