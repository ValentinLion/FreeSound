package treejson;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.json.JSONArray;
import org.json.JSONObject;

/**
 *
 * @author karim oubah
 */
public class Version2 {

    static Map<String, List<String>> mapChild;
    static Map<String, String> mapName;
    static Map<String, String> mapJSON;
    static Map<String, Float> mapStats;
    static String json;
    static int sum = 0;

    public static void main(String[] args) throws IOException {
        mapChild = new HashMap<>();
        mapName = new HashMap<>();
        mapJSON = new HashMap<>();
        mapStats = new HashMap<>();

        BufferedReader br = new BufferedReader(new FileReader("stats.csv"));
        String line = br.readLine();

        while (line != null) {
            mapStats.put(line.substring(0, line.indexOf(',')), new Float(line.substring(line.indexOf(',') + 1)));
            sum += new Float(line.substring(line.indexOf(',') + 1));
            line = br.readLine();
        }
        br.close();

        for (Map.Entry<String, Float> entry : mapStats.entrySet()) {
            mapStats.put(entry.getKey(), entry.getValue() / sum * 100);
        }
        System.out.println(mapStats);
        System.out.println(sum);

        br = new BufferedReader(new FileReader("ontology.json"));
        StringBuilder sb = new StringBuilder();
        line = br.readLine();

        while (line != null) {
            sb.append(line);
            sb.append(System.lineSeparator());
            line = br.readLine();
        }
        br.close();

        String everything = sb.toString();
        JSONArray array = new JSONArray(everything);
        JSONObject jsonTemp;
        JSONArray arrTemp;
        for (int i = 0; i < array.length(); i++) {
            jsonTemp = array.getJSONObject(i);
            if (!mapName.containsKey(jsonTemp.getString("id"))) {
                mapName.put(jsonTemp.getString("id"), jsonTemp.getString("name"));

            }
            if (!mapChild.containsKey(jsonTemp.getString("id"))) {
                mapChild.put(jsonTemp.getString("id"), new ArrayList<>());

            }
            arrTemp = jsonTemp.getJSONArray("child_ids");
            for (int j = 0; j < arrTemp.length(); j++) {
                mapChild.get(jsonTemp.getString("id")).add(arrTemp.getString(j));
            }
        }

        StringBuilder tempJSON;
        String name;
        for (Map.Entry<String, List<String>> entry : mapChild.entrySet()) {
            tempJSON = new StringBuilder();
            name = mapName.get(entry.getKey());
            if (mapStats.get(name) == null) {
                tempJSON.append("{\n \"name\": \"" + name + " 0%" + "\", \"children\": [");
            } else {
                tempJSON.append("{\n \"name\": \"" + name + " " + mapStats.get(name) + " %" + "\", \"children\": [");
            }
            for (int i = 0; i < entry.getValue().size(); i++) {
                name = mapName.get(entry.getValue().get(i));
                if (mapStats.get(name) == null) {
                    tempJSON.append("{\"name\": \"" + name + " 0%" + "\"},");

                } else {
                    tempJSON.append("{\"name\": \"" + name + " " + mapStats.get(mapName.get(entry.getKey())) + " %" + "\"},");
                }
            }
            if (tempJSON.charAt(tempJSON.length() - 1) == ',') {
                tempJSON.deleteCharAt(tempJSON.length() - 1);
            }
            tempJSON.append("],");
            if (tempJSON.charAt(tempJSON.length() - 1) == ',') {
                tempJSON.deleteCharAt(tempJSON.length() - 1);
            }
            tempJSON.append("},");
            mapJSON.put(mapName.get(entry.getKey()), tempJSON.toString());
        }
        //System.out.println(mapChild.get("/m/0dgw9r"));

        StringBuilder globalJSON = new StringBuilder();
        globalJSON.append("{ \"name\": \"Ontology 100%\", \"children\": [");
        json = "";
        json += "{ \"name\": \"Ontology\", \"children\": [";
        json += "]}";

        for (Map.Entry<String, String> entry : mapJSON.entrySet()) {
            //System.out.println(entry.getValue());
            globalJSON.append(entry.getValue());
        }
        if (globalJSON.charAt(globalJSON.length() - 1) == ',') {
            globalJSON.deleteCharAt(globalJSON.length() - 1);
        }
        globalJSON.append("]}");
        System.out.println(globalJSON.toString());
    }

}
