import java.io.File;
import jfml.JFML;
import jfml.FuzzyInferenceSystem;
import jfml.knowledgebase.variable.KnowledgeBaseVariable;

public class EvaluateXML
{

  private static String xml_filename = "test_jfml_out.xml";
  private static FuzzyInferenceSystem fis;

  private static float evaluate(float angle, float change)
  {
    // Set input values:
    KnowledgeBaseVariable var_angle = fis.getVariable("Angle");
    KnowledgeBaseVariable var_change = fis.getVariable("ChangeAngle");
    var_angle.setValue(angle);
    var_change.setValue(change);
    // Perform fuzzy inference:
    fis.evaluate();
    // Get output value:
    KnowledgeBaseVariable var_force = fis.getVariable("Force");
    float force = var_force.getValue();
    return force;
  }
  
  public static void main(String[] args)
  {
    float angle = 10.0f;
    float change = 0.0f;
    if (args.length == 2) {
      angle = Float.parseFloat(args[0]);
      change = Float.parseFloat(args[1]);
    }
    System.out.println("Loading Fuzzy System from XML file " + xml_filename);
    fis = JFML.load(new File(xml_filename));
    System.out.println(fis.toString());

    System.out.printf("Inputs: angle=%f, change=%f", angle, change);
    float force = evaluate(angle, change);
    System.out.printf(" => force=%f\n", force);
  }
}
