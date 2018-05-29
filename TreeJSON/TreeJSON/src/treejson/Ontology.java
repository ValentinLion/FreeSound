/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package treejson;

import java.util.Objects;

/**
 *
 * @author 14003909
 */
public class Ontology {
    
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

