from collections import defaultdict as ddict
from nltk.corpus import wordnet as wn
import joblib
import copy
import numpy as np


# exclude random domains
def read_types(file_name):
    f = open(file_name)
    canonical_to_original = {}
    garbage_types = set()
    domains = set()
    bad_domains = set(["freebase", "atom", "dataworld", "apps", "schema", "scheme", "topics", "domain"])

    for line in f:
        canonical, domain, original, _ = line.strip().split("\t")
        if domain in bad_domains:
            garbage_types.add(canonical)


        domains.add(domain)    
        canonical_to_original[canonical] = original

    print(domains)    
    return canonical_to_original, garbage_types


# very horrible code for reading alignments_all_annotated.txt
def read_annotated_file(file_name):
    f = open(file_name)
    curr_type = ''
    wordnet = False

    taxonomy = {}
    garbage_types = set()
    wordnet_types = set()
    wn_to_fb = {}

    for line in f:
        line = line.strip()
        if line == '':
            wordnet = False
            continue
        elif line == "======================":
            wordnet = True

        elif line.startswith("examples"):
            continue

        elif not wordnet:
            if '<-' in line:
                curr_type, fb_parent = line.split('(<-')
                fb_parent = fb_parent[:-1].lstrip()
                curr_type = curr_type.strip()

                taxonomy[curr_type] = ddict(list)
                taxonomy[curr_type]["parent"].append(fb_parent)

            elif '->' in line:
                curr_type, fb_child = line.split('(->')
                fb_child = fb_child[:-1].lstrip()
                curr_type = curr_type.strip()
                taxonomy[curr_type] = ddict(list)
                taxonomy[curr_type]["child"].append(fb_child)


            else:
                curr_type = line
                taxonomy[curr_type] = ddict(list)

            if line.startswith("(X)"):
                garbage_types.add(curr_type)

        elif line.startswith("*"):
            wn_type = wn.synset(line[1:].split(":")[0])
            taxonomy[curr_type]["exact"].append(wn_type)
            wn_to_fb[wn_type] = curr_type

            wordnet_types.add(wn_type)

        elif line.startswith("^"):
            wn_type = wn.synset(line[1:].split(":")[0])
            taxonomy[curr_type]["parent"].append(wn_type)

            wordnet_types.add(wn_type)

        elif line.startswith("$"):
            wn_type = wn.synset(line[1:].split(":")[0])
            taxonomy[curr_type]["child"].append(wn_type)

            wordnet_types.add(wn_type)



    keys = taxonomy.keys()

    for key in keys:
        if key in garbage_types:
            del taxonomy[key]


    keys = wn_to_fb.keys()

    for key in keys:
        val = wn_to_fb[key]
        if val in garbage_types:
            del wn_to_fb[key]


    return taxonomy, wordnet_types, wn_to_fb


def get_path(curr_type, taxonomy, all_nodes, node_paths, processed):
    # a freebase type

    if curr_type in processed:
        return


    processed.add(curr_type)
    if curr_type in taxonomy:
        if "child" in taxonomy[curr_type]:
            all_children = taxonomy[curr_type]["child"]
            for child in all_children:
                if child in wn_to_fb:
                    child = wn_to_fb[child]

                node_paths[child].add(curr_type)
                get_path(child, taxonomy, all_nodes, node_paths, processed)



    if curr_type in taxonomy:
        if "exact" in taxonomy[curr_type]:
            wn_type = taxonomy[curr_type]["exact"][0]
            all_parents_curr = wn_type.hypernyms()

        elif "parent" in taxonomy[curr_type]:
            all_parents_curr = taxonomy[curr_type]["parent"]

        else:
            return

    else:
        all_parents_curr = curr_type.hypernyms()



    # replace with exact freebase if present
    for parent in all_parents_curr:
        if parent in wn_to_fb:
            parent = wn_to_fb[parent]

        all_nodes.add(parent)
        node_paths[curr_type].add(parent)

        get_path(parent, taxonomy, all_nodes, node_paths, processed)




def create_typenet(taxonomy, wn_to_fb, canonical_to_original, garbage_types):
    '''
        Function to create the entire type-net tree with freebase types in their right places.
    '''
    # == all the nodes in the dataset
    all_nodes = set()
    node_paths = ddict(set) 
    processed = set()

    for freebase_type in taxonomy:
        if freebase_type in garbage_types:
            continue
        all_nodes.add(freebase_type)
        get_path(freebase_type, taxonomy, all_nodes, node_paths, processed)



    f = open("typenet_structure.txt", "w")
    all_edges = set()
    all_types = set()

    for node in node_paths:
        if node in garbage_types:
            continue

        for parent in node_paths[node]:
            if parent in garbage_types:
                continue

            if node in canonical_to_original:
                node = canonical_to_original[node]

            if parent in canonical_to_original:
                parent = canonical_to_original[parent]

            if (node, parent) not in all_edges:
                f.write("%s -> %s\n" %(node, parent))
                all_edges.add((node, parent))
                # convert Synset objects to their string representations
                all_types.add("%s" %parent)
                all_types.add("%s" %node)


    f.close()
    return all_types



def write_annotations(taxonomy, garbage_types, canonical_to_original):

    f = open("typenet_annotations.txt", "w")
    total_types = 0
    for freebase_type in taxonomy:
        if freebase_type in garbage_types:
            continue

        assert(len(taxonomy[freebase_type]) != 0)
    

        f.write("%s\n" %canonical_to_original[freebase_type])
        for relation in taxonomy[freebase_type]:
            for element in taxonomy[freebase_type][relation]:
                if element in canonical_to_original:
                    f.write("%s:%s\n" %(relation, canonical_to_original[element]))
                else:
                    f.write("%s:%s\n" %(relation, element))

        f.write("\n")
        total_types += 1

    print("Total types: %d" %total_types)
    return




# == Code for adding new links and checking if resulting structure is still a DAG.


def dfs(node, ancestor, adj_matrix, transitive_closure):
    if ancestor != node:
        transitive_closure[node][ancestor] = 1.0

    for _parent in xrange(len(adj_matrix[ancestor])):
        if adj_matrix[ancestor][_parent]:
            dfs(node, _parent, adj_matrix, transitive_closure)

    return 

def run_transitive_closure(adj_matrix):
    num_nodes = len(adj_matrix)
    transitive_closure = copy.deepcopy(adj_matrix)
    for node in xrange(num_nodes):
        dfs(node, node, adj_matrix, transitive_closure)

    return transitive_closure



def check_dag2(adj_matrix):

    def dfs_dag(node, visited):
        if node in visited:
            return False

        visited.add(node)

        ret_val = True
        for parent in xrange(len(adj_matrix[node])):
            if adj_matrix[node][parent]:
                ret_val &= dfs_dag(parent, visited)

        # backtrack
        visited.remove(node)
        return ret_val


    is_dag = True
    visited = set()

    for node in xrange(len(adj_matrix)):
        # launch a dfs 

        if node not in visited:
            curr_dag = dfs_dag(node, visited)
            if not curr_dag:
                _node = inv_type_dict[node]
                print(_node)
        
            is_dag &= curr_dag

    return is_dag



def add_links_from_freebase(fname1, fname2, type_dict):
    f = open(fname1)

    adj_matrix = np.zeros((len(type_dict), len(type_dict)))

    for line in f:
        node, _, parent = line.strip().split(" ")
        adj_matrix[type_dict[node]][type_dict[parent]] = 1.0

    transitive_closure = run_transitive_closure(adj_matrix)

    f = open(fname2)
    added = 0
    for line in f:
        line = line.strip()
        if line.startswith("x") or len(line) == 0:
            continue



        node, _, parent, _, _ = line.strip().split(" ")

        node = type_dict[node]
        parent = type_dict[parent]

        if not transitive_closure[node][parent]:
            added += 1
            adj_matrix[node][parent] = 1.0


    # check if the graph represented by this adjacency matrix is a DAG
    assert(check_dag2(adj_matrix))


    print("Added %d new Freebase -> Freebase links" %added)


    f = open(fname1, "w")


    #traversed_edges = set()

    leaf_nodes = adj_matrix.T.sum(axis=-1)

    def _dfs(node, level = 0):
        for parent in xrange(len(adj_matrix[node])):
            if adj_matrix[node][parent]:
                #traversed_edges.add((node, parent))
                f.write("%s%s -> %s\n" %(" "*level, inv_type_dict[node], inv_type_dict[parent]))
                _dfs(parent, level+1)


    for node in xrange(len(adj_matrix)):
        # only do dfs for leaf nodes
        if leaf_nodes[node] == 0:
            _dfs(node)
            f.write("\n")

    f.close()

    transitive_closure_final = run_transitive_closure(adj_matrix)

    return transitive_closure_final


if __name__ == "__main__":
    taxonomy, wordnet_types, wn_to_fb = read_annotated_file("alignments_all_annotated.txt")

    print("Total number of wordnet types in direct relation: %d" %len(wordnet_types))
    canonical_to_original, garbage_types = read_types("cleaned_types")

    # write the annotations 
    write_annotations(taxonomy, garbage_types, canonical_to_original)

    # create the typeNet taxonomy from the alignments
    all_types = create_typenet(taxonomy, wn_to_fb, canonical_to_original, garbage_types)

    all_types.add("NO_TYPES")
    type_dict = { _type : idx for (idx, _type) in enumerate(all_types)}
    inv_type_dict = {idx : _type for (_type, idx) in type_dict.iteritems()}

    transitive_closure = add_links_from_freebase('typenet_structure.txt', 'conditional_freebase_links.txt', type_dict)
    joblib.dump(transitive_closure, "TypeNet_transitive_closure.joblib")

    joblib.dump(type_dict, "TypeNet_type2idx.joblib")
    print("number types: %d" %len(type_dict))
