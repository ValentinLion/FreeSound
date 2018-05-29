package treejson;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import org.json.JSONArray;
import org.json.JSONObject;

/**
 *
 * @author karim oubah
 */
public class TreeJSON {

    static Map<String, List<String>> mapChild;
    static Map<String, String> mapName;
    static Map<String, String> mapJSON;
    static String json;

    public static void main(String[] args) throws IOException {
        mapChild = new HashMap<>();
        mapName = new HashMap<>();
        mapJSON = new HashMap<>();
        BufferedReader br = new BufferedReader(new FileReader("ontology.json"));
        StringBuilder sb = new StringBuilder();
        String line = br.readLine();

        while (line != null) {
            sb.append(line);
            sb.append(System.lineSeparator());
            line = br.readLine();
        }
        br.close();

        String everything = sb.toString();
        //System.out.println(everything);
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
        for (Map.Entry<String, List<String>> entry : mapChild.entrySet()) {
            tempJSON = new StringBuilder();
            tempJSON.append("{\n \"name\": \"" + mapName.get(entry.getKey()) + "\", \"children\": [");
            for (int i = 0; i < entry.getValue().size(); i++) {
                tempJSON.append("{\"name\": \"" + mapName.get(entry.getValue().get(i)) + "\"},");
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
        globalJSON.append("{ \"name\": \"Ontology\", \"children\": [");
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

    public static void parcours(String id) {
        for (int i = 0; i < mapChild.get(id).size(); i++) {
            parcours(mapChild.get(id).get(i));
        }
    }

    public static String recursif(String s) {
        StringBuilder sb = new StringBuilder(s);
        sb.append("ee");
        for (Map.Entry<String, List<String>> entry : mapChild.entrySet()) {
            json += "{\n \"name\": \"" + entry.getKey() + "\", \"children\": [";
            parcours(entry.getKey());
            recursif(sb.toString());
            System.out.println(entry.getKey() + "/" + entry.getValue());
            json += "]";
        }

        return sb.toString();
    }

    public static String buildJSON() {
        return "";
    }

}

class Ontology {

    private String id, name;

    public Ontology(String id, String name) {
        this.id = id;
        this.name = name;
    }

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    @Override
    public int hashCode() {
        int hash = 7;
        return hash;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj == null) {
            return false;
        }
        if (getClass() != obj.getClass()) {
            return false;
        }
        final Ontology other = (Ontology) obj;
        if (!Objects.equals(this.id, other.id)) {
            return false;
        }
        return true;
    }

}
