#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 11:53:11 2022

@author: mpalermo
"""

import copy
from dataclasses import dataclass
import time
import datetime
from typing import Any, Type, Callable, Optional
from collections.abc import KeysView  # for type-hinting purpose only
from pubchemtools.communication import HTTP_request, PUBCHEM_load_balancer
from pubchemtools.ghs_ranking import EmptyRanking, GES002
import rdkit
from rdkit import Chem


@dataclass
class ChemicalIDs:
    """Contains textual chemical identifiers of a Molecule.

    A dataclass that contains all the chemical identifiers for a given molecule.

    Attributes
    ----------
    InChIKey : str, optional
        InChI-Key associated with the compound. The default is None.
    InChI : str, optional
        InChI representation of the compound. The default is None.
    SMILES : str, optional
        SMILES representation of the compound. The default is None.
    IUPAC : str, optional
        Standard IUPAC chemical name of the compound. The default is None.
    name : str, optional
        Traditional (or other) chemical name of the compound. The default is None.
    pubchem_cid : int
        Pubchem Compound IDentifier (CID)
    sdf : str, optional
        SDF string containing the molecular structure
    mol : str, optional
        MOL string containing the molecular structure
    """

    InChIKey: Optional[str] = None
    InChI: Optional[str] = None
    IUPAC: Optional[str] = None
    name: Optional[str] = None
    SMILES: Optional[str] = None
    pubchem_cid: Optional[int] = None


@dataclass
class GHSReferences:
    """Contains regulatory data fetched from PubChem database.

    The companies and notification fields are valid only for ECHA references.

    Attributes
    ----------
    reference_id : int, optional
        Numerical ID of the regulatory reference assigned by PubChem
    source_name : str, optional
        Name of the entity that reported the regulatory profile (e.g. ECHA)
    hazard_codes : [str], optional
        Hazard P(recautionary) statements of the molecule
    companies : int, optional
        Number of companies that provided a notification to ECHA
    notifications : int, optional
        Total number of companies that provided the same notification to ECHA
    """

    reference_id: Optional[int] = None
    source_name: Optional[str] = None
    hazard_codes: Optional[list[str]] = None
    companies: Optional[int] = None
    notifications: Optional[int] = None


@dataclass
class UserProperties:
    """Dataclass containing user properties."""

    @property
    def count(self):
        """Return the number of the compound user properties.

        Returns
        -------
        int
            Number of user properties.
        """
        return len(self.__dict__)

    def __setattr__(self, name: str, value: Any, class_type: Type = object) -> None:
        """Modification of __setattr__ to prevent user to add a new attribute.

        This modification to the __setattr__ method forces the user to use
        the add_property() method of the Compound class to add a new property.

        Parameters
        ----------
        name : str
            Name of the variable.
        value : typing.Any
            Value of the variable.
        class_type : typing.ClassVar, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        # Forbids creating a new property through direct variable assignation
        # except the variable already exists in UserProperties
        if class_type is not Compound:  # runs only if called from Compound instance
            if name in self.__dict__:
                super().__setattr__(name, value)
            else:
                print("To set a new property, use the add_property instance method")
                return
        else:
            super().__setattr__(name, value)

    def __repr__(self):
        """Return a string representation of the Dataclass.

        Returns
        -------
        str
            String representation of the dataclass.

        """
        # For some reason, the modification of the __setattr__ method breaks
        # dataclasses automatic __repr__, so I'm reimplementing it
        return str(self.__dict__)


GES002_inst: GES002 = GES002()
"""GES002 : module level instance of GES002 regulatory profile

   It will be used by all istances of the Compound class to compute the GHS
   ranking of the molecule using ECHA hazard phrases.
"""


class Compound:
    """Representation of a molecule and its related data.

    Depending on the user input, the istance might be initialized either via
    a chemical identifier or structure data. In the first case,
    chemical_IDs will be set to the user input, and the molecule attribute
    will be left empty. If structure data is provided through the sdf/mol
    filepath argument or a rdkit mol object, an attempt to generate the InChI-Key through
    RDkit will be performed (though it might not always succeed).

    Parameters
    ----------
    InChIKey : str, optional
        InChI-Key associated with the compound. The default is None.
    InChI : str, optional
        InChI representation of the compound. The default is None.
    SMILES : str, optional
        SMILES representation of the compound. The default is None.
    IUPAC : str, optional
        Standard IUPAC chemical name of the compound. The default is None.
    name : str, optional
        Traditional (or other) chemical name of the compound. The default is None.
    sdf : str, optional
        Filepath of an SDF file containing a single structure. If an SDF
        file with multiple molecules is provided, only the first one will
        be considered. The molecular structure will be stored as an rdkit
        mol object. The default is None.
    mol : str, optional
        Filepath of a MOL file. The molecular structure will be stored as an rdkit
        mol object. The default is None.
    rdkitmol : rdkit.Chem.rdchem.Mol, optional
        RDKit molecule object. The default is None.
    autofetch : TYPE, optional
        If set to True, an attempt to retrieve chemical IDs, vendors and GHS
        data will be performed upon instance initialization. The default is True.

    Attributes
    ----------
    excluded_properties : [str]
        (class attribute) List of attributes that won't be written to a file if the compound
        is exported through a library instance.

    builtin_properties : [str]
        (class attribute) List of attributes that will be defined in the __init__ method and
        won't be considered as a user property.
    vendors : [str]
        List of vendors supplying the compound. None if no data was fetched through Pubchem.
    molecule : rdkit.Chem.rdchem.Mol
        rdkit mol object representing the compound. None if structure data is not available.
    compound_in_PUBCHEm : bool
        If True, the compound was found on Pubchem. Defaults to None if no Pubchem search
        has been performed.
    ghs_references : dict
        Contains one or more set of regulatory data. Data is saved as GHSReferences
        dataclass.
    user_properties : UserProperties
        Dataclass containing user properties that have been added to the compound instance
        through the add_property method.

    Raises
    ------
    ValueError
        If no input parameter is given upon object creation, raises ValueError.

    Returns
    -------
    None.
    """

    excluded_properties = ["_fetch_pairs"]
    builtin_properties = [
        "chemical_ids",
        "_vendors_data",
        "_vendors",
        "molecule",
        "_user_properties",
        "compound_in_PUBCHEM",
        "ghs_references",
        "_fetch_pairs",
        "_linked_libraries",
    ]

    def __init__(
        self,
        InChIKey: Optional[str] = None,
        InChI: Optional[str] = None,
        SMILES: Optional[str] = None,
        IUPAC: Optional[str] = None,
        name: Optional[str] = None,
        sdf: Optional[str] = None,
        mol: Optional[str] = None,
        rdkitmol: Optional[rdkit.Chem.rdchem.Mol] = None,
        autofetch: Optional[bool] = True,
    ) -> None:
        """Initialize the Compound class."""
        # Instance attributes
        self._vendors_data: list[dict]
        self._vendors: Optional[set[str]] = None  # DEV: implement as a property
        self.molecule: Optional[rdkit.Chem.rdchem.Mol] = None
        self._user_properties: UserProperties = UserProperties()  # implement as a property
        self.compound_in_PUBCHEM: Optional[bool] = None
        self.ghs_references: dict[int, GHSReferences] = {}
        self._linked_libraries: list[Library] = []
        self.chemical_ids: ChemicalIDs

        # DEV: should be in try/except
        # Initialize differently depending on user input
        if sdf is not None:
            suppl: rdkit.Chem.SDMolSupplier = Chem.SDMolSupplier(sdf)
            self.molecule = suppl[0]
        if mol is not None:
            self.molecule = Chem.MolFromMolFile(mol)
        if rdkitmol is not None:
            self.molecule = rdkitmol
        if any([InChIKey, InChI, IUPAC, name, SMILES, sdf, mol, rdkitmol]):
            self.chemical_ids = ChemicalIDs(
                InChI=InChI,
                InChIKey=InChIKey,
                IUPAC=IUPAC,
                name=name,
                SMILES=SMILES,
            )
        else:
            raise ValueError(
                "Compound object initialization requires at least one argument"
            )

        # Couples functions that supply Pubchem url to functions that set the data returned
        # by Pubchem. This allow to fetch Pubchem data all in one go and worry about
        # setting the output to the correct attributes later
        self._fetch_pairs: dict[Callable, Callable] = {
            self._vendors_url: self._set_vendors_pubresponse,
            self._ghs_url: self._set_ghs_pubresponse,
        }

        if autofetch:
            self.fetch_data()

        # Without an InChIKey (or other chemical IDs) Pubchem search won't work
        # if self.molecule:
        #     self.chemical_ids.InChIKey = Chem.MolToInchiKey(self.molecule)

    def _chemical_ids_url(self) -> tuple[str, Optional[str]]:
        """Generate Pubchem REST url to retrieve InChI,InChiKey, SMILES,IUPACName.

        Returns
        -------
        tuple[str, str]
            Pubchem REST url and post data.

        """
        # DEV: should add a try/except block here
        url: str
        post: Optional[str] = None  # None in case no post data is needed e.g. InChIKey
        # DEV: also SMILES and IUPAC should be implemented with post as they can get long

        if self.chemical_ids.InChIKey is not None:
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{self.chemical_ids.InChIKey}/property/InChI,IsomericSMILES,IUPACName/JSON"
        elif self.chemical_ids.InChI is not None:
            url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchi/property/InChIKey,IsomericSMILES,IUPACName/JSON"
            post = f"inchi={self.chemical_ids.InChI}"
        elif self.chemical_ids.SMILES is not None:
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{self.chemical_ids.SMILES}/property/InChIKey,InChI,IUPACName/JSON"
        elif self.chemical_ids.IUPAC is not None:
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/iupacname/{self.chemical_ids.IUPAC}/property/InChIKey,InChI,IsomericSMILES/JSON"
        elif self.chemical_ids.name is not None:
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{self.chemical_ids.name}/property/InChIKey,InChI,IsomericSMILES,IUPACName/JSON"
        else:
            url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/sdf/property/InChI,InChIKey,IsomericSMILES,IUPACName/JSON"
            molblock = Chem.MolToMolBlock(self.molecule)
            post = {"sdf": molblock}

        return url, post

    def _vendors_url(self) -> tuple[str, Optional[str]]:
        """Generate Pubchem REST url to retrieve vendors data.

        Returns
        -------
        tuple[str, str]
            Pubchem REST url and post data.

        """
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/categories/compound/{self.chemical_ids.pubchem_cid}/JSON/"
        post = None
        return url, post

    def _ghs_url(self) -> tuple[str, Optional[str]]:
        """Generate Pubchem REST url to retrieve GHS data.

        Returns
        -------
        tuple[str, str]
            Pubchem REST url and post data.

        """
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{self.chemical_ids.pubchem_cid}/JSON/?heading=GHS+Classification"
        post = None
        return url, post

    def _set_chemical_ids_pubresponse(self, response: Any) -> None:
        """Set chemical IDs from data returned by Pubchem.

        First the function identifies which was the input chemical ID that was supplied to
        the Compound constructor, then proceeds setting the other missing chemical IDs
        retrieved by Pubchem. Note: SMILES representation is stored, within the ChemicalIDs
        dataclass, in a attribute named "SMILES". Since the function is checking against
        attribute names to identified which ID was supplied by the user, and since Pubchem
        has two SMILES repr. (canonical and isomeric), we need to rename the identifier to
        correctly retrieve PubChem SMILES.

        Parameters
        ----------
        response : Any
            Contains the Pubchem REST API response in JSON format.

        Returns
        -------
        None.

        """
        identifiers: dict[str, str] = self.chemical_ids.__dict__
        chosen: dict[str, str] = {}

        # relies on definition order of dict - may be unreliable depending
        # on python base library implementation
        for identifier, value in identifiers.items():
            if value is not None:
                if identifier == "SMILES":  # in case we initialized Compound with a SMILES
                    # Pubchem distinguishes between isomeric and canonical SMILES. We always
                    # retrieve the Isomeric, so we set here the right label to match data
                    identifier = "IsomericSMILES"
                chosen = {identifier: value}

        if response is None:
            print(
                f"No Chemical IDs were found for {list(chosen.keys())[0]}: "
                + "{list(chosen.values())[0]}"
            )
        else:
            # we merge user input with chemical ids retrieved by pubchem
            data = response["PropertyTable"]["Properties"][0] | chosen

            self.chemical_ids.pubchem_cid = data["CID"]
            self.chemical_ids.InChIKey = data["InChIKey"]
            self.chemical_ids.InChI = data["InChI"]
            self.chemical_ids.SMILES = data["IsomericSMILES"]
            self.chemical_ids.IUPAC = data["IUPACName"]

            # Create Compound rdkit mol
            self.molecule = Chem.MolFromInchi(self.chemical_ids.InChI)

    def _set_vendors_pubresponse(self, response: Any) -> None:
        """Set vendors information from data returned by Pubchem.

        Currently we do nothing more than store the data, but more functionality should
        be implemented in the future.

        Parameters
        ----------
        response : Any
            Contains the Pubchem REST API response in JSON format.

        Returns
        -------
        None.

        """
        self._vendors_data = response["SourceCategories"]["Categories"][0]["Sources"]

    def _set_ghs_pubresponse(self, response) -> None:
        """Set GHS information from data returned by Pubchem.

        The function takes Pubchem GHS data and stores it in one or more GHSReference
        instances (depending on the number of records on Pubchem). Individual hazard phrases
        are extracted and stored in a list. In case of data from
        ECHA, the number of companies and notifications is also extracted. Finally, a GHS
        score is calculated using the scheme defined in pubchemtools.ghs_ranking using, as
        default, the regulatory profile GES002.

        Parameters
        ----------
        response : Any
            Contains the Pubchem REST API response in JSON format.

        Returns
        -------
        None.

        """
        # data from Pubchem
        ref_sources: list[dict] = response["Record"]["Reference"]
        ref_data: list[dict] = response["Record"]["Section"][0]["Section"][0]["Section"][0][
            "Information"
        ]

        source: dict

        # Collect the referenced ids and source from the ["Record"]["Reference"] header
        for source in ref_sources:
            ref_id: int = source["ReferenceNumber"]
            source_name: str = source["SourceName"]
            self.ghs_references[ref_id] = GHSReferences(
                reference_id=ref_id, source_name=source_name
            )

        # Then, loop through the H-phrases and link them to the ref_id of the source
        ref: dict

        for ref in ref_data:
            ref_id = ref["ReferenceNumber"]
            ref_name: str = ref["Name"]

            if ref_name == "GHS Hazard Statements":
                # print('\n \x1b[1;34;80mGHS Hazard Statements\x1b[1;34;0m \n')
                hazard_codes: list[str] = []
                entry: dict[str, str]
                for entry in ref["Value"]["StringWithMarkup"]:
                    hphrase: str = entry["String"][0:4]
                    if hphrase == "Not ":
                        continue
                    if hphrase == "Repo":
                        self.ghs_references[ref_id].companies = int(
                            entry["String"].split()[8]
                        )
                        hazard_codes = ["Not Hazardous"]
                    else:
                        hazard_codes.append(entry["String"][0:4])
                self.ghs_references[ref_id].hazard_codes = hazard_codes
            else:
                # just ignore additional stuff like "Signal", Pictograms(s), etc
                pass

            # If the reference is from ECHA, lets extract companies and notification to
            # identify the most reliable source
            if ref_name == "ECHA C&L Notifications Summary":
                # print('\n \x1b[1;32;80mECHA HERE\x1b[1;32;0m \n')
                # print(ref)
                echa_entry: str = ref["Value"]["StringWithMarkup"][0]["String"]
                splitted_entry: list[str] = echa_entry.split()
                self.ghs_references[ref_id].companies = int(splitted_entry[5])
                self.ghs_references[ref_id].notifications = int(splitted_entry[8])

        self.calculate_ghs_ranking()  # at last...

    def calculate_ghs_ranking(self, ranking_profile: EmptyRanking = GES002_inst) -> None:
        """Compute the compound GHS ranking.

        Ranking is calculated using a user defined regulatory profile. Each profile contains
        a score for each hazard phrase, and the final score of a compound is the sum of the
        scores of all its h-phrases. The default ranking profile is GES002 which takes into
        account only European Chemicals Agency (ECHA) references.

        Parameters
        ----------
        ranking_profile : RegProfile, optional
            Regulatory profile. The default is GES002_instance.

        Returns
        -------
        None

        """
        ranking_name: str = type(ranking_profile).__name__ + "_GHS_ranking"
        score: int = ranking_profile.rank(self.ghs_references)

        self.add_property(ranking_name, score)

    def fetch_data(self) -> None:
        """Retrieve data from Pubchem (single thread I/O).

        The function generates the url and post data and submits them to the Pubchem REST
        API. Then it retrieve Pubchem results and assign the results to the respective
        property.

        Returns
        -------
        None

        """
        # Chemical IDs
        url: str
        post: Optional[str]
        response: Any  # it's a JSON

        url, post = self._chemical_ids_url()
        response = HTTP_request(url, post)
        self._set_chemical_ids_pubresponse(response)

        url, post = self._vendors_url()
        response = HTTP_request(url, post)
        self._set_vendors_pubresponse(response)

        url, post = self._ghs_url()
        response = HTTP_request(url, post)
        self._set_ghs_pubresponse(response)

    @property
    def vendors(self) -> set[str]:
        """Return a list of vendors of the compound.

        Returns
        -------
        set[str]
            A list of the vendors names of the compound.

        """
        if not hasattr(self, "_vendors_data"):
            return set()

        return {vendor_entry["SourceName"] for vendor_entry in self._vendors_data}

    @property
    def user_properties(self) -> UserProperties:
        """Return all user defined properties.

        Returns
        -------
        UserProperties
            A dataclass containing all user defined properties.

        """
        return self._user_properties

    def save(self, filename: str, file_format: str = "sdf") -> None:
        """Save the molecule to a file.

        Parameters
        ----------
        filename : str
            Filename.
        file_format : str, optional
            Format of output file. The default is "sdf".

        Returns
        -------
        None.

        """
        # DEV: should add restrains to arguments for file_format
        if self.molecule is None:
            raise UnboundLocalError(
                "No molecule object is present in Compound. " + "Cannot save to file."
            )

        writer: Chem.SDWriter = Chem.SDWriter(filename + "." + file_format)

        molecule = copy.copy(self.molecule)
        molecule.SetProp("InChI", self.chemical_ids.InChI)
        molecule.SetProp("InChIKey", self.chemical_ids.InChIKey)
        molecule.SetProp("SMILES", self.chemical_ids.SMILES)
        molecule.SetProp("IUPAC", self.chemical_ids.IUPAC)

        molecule.SetProp("Vendors", ",".join(self.vendors))

        writer.write(molecule)

    def add_property(self, property_name: str, property_value: Any) -> None:
        """Add a property to the Compound user_properties Dataclass instance.

        The use of the add_property method prevents the user from inadvertly polluting
        the attributes of the Compound instances.

        Parameters
        ----------
        property_name : str
            Name of the user-defined property.
        property_value : Any
            Value of the user-defined property.

        Returns
        -------
        None

        """
        if property_name not in self._user_properties.__dict__:
            self._user_properties.__setattr__(property_name, property_value, type(self))

            # after adding the property to the compound instance, register it to all the
            # libraries containing the compound
            # an event based system would be better, but for now this will suffice
            for library in self._linked_libraries:
                library._register_property(property_name)
        else:
            # if the property already exists in the Compound instance, then modify its value
            self._user_properties.__setattr__(property_name, property_value, type(self))

    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent user to edit Compound instance properties.

        The function still allows users to modify built_in properties such as GHS data,
        vendors etc. This will be prevented in future releases through implementing data
        access through property decorators.

        Parameters
        ----------
        name : str
            Name of the property to be added.
        value : Any
            Value of the property to be added.

        Raises
        ------
        AttributeError
            AttributeError is raised if the variable does not alredy exists in the Compound
            instance or if is not among the builtin properties defined in the class
            attribute.

        Returns
        -------
        None
        """
        if name not in self.__dict__ and name not in self.builtin_properties:
            raise AttributeError(
                "Assigning a property with a = operator is forbidden. Please use the add_property instance method instead. "
                + "\n"
                + f' --> compound.add_property("{name}", {value})',
            )

        super().__setattr__(name, value)


class Library:
    """Library for managing multiple compounds objects.

    The library class allows the user to store data of multiple Compoud
    objects e.g. retrieving PUBCHEM information for all objects at once
    and to read or export libraries of compound from or to file.

    Parameters
    ----------
    compounds_list : list, default None
        A list containing Compounds objects.

    Attributes
    ----------
    compounds_list : list
        A list containing Compounds objects.

    """

    def __init__(self, compounds_list: Optional[list[Compound]] = None) -> None:
        """Initialize Library object."""
        self._registered_properties: set[str] = set()

        if compounds_list is None:
            self.compounds_list = []  # list as default argumetn wreaks havoc
        else:
            self.compounds_list = compounds_list

        for compound in self.compounds_list:
            compound._linked_libraries.append(self)

    def fetch_data(self, attempts: int = 3) -> None:
        """Retrieve compounds data from PUBCHEM database.

        The function creates all urls to retrieve chemical IDs, GHS and vendors
        data from PUBCHEM, then feed them to the PUBCHEM_load_balancer function
        from communication, and then sets the properties to each Compound
        object.


        Parameters
        ----------
        attempts : TYPE, optional
            Number of attempts to be tried by PUBCHEM_load_balancer in case
            communication with PUGREST fails. The default is 3.

        Returns
        -------
        None.

        """
        # for compound in self.compounds_list:
        #     # using function as dict key to pair url request and setter!
        #     for url_fetcher, setter in compound._fetch_pairs.items():
        #         url, post = url_fetcher()
        #         response = HTTP_request(url, post)
        #         setter(response)
        initial_time: float = time.time()

        url_requests: list[Callable] = []

        print("Fetching chemical IDs")
        # first, we need a CID...
        cid_requests: list[Callable] = []
        for compound in self.compounds_list:
            cid_requests.append(compound._chemical_ids_url)

        # lets grab the chemical IDs from pubchem...
        # data contains a dict of JSONs
        data: dict[Callable, Any] = PUBCHEM_load_balancer(cid_requests, attempts)

        # ... and set them in each compound object. If no InChIKey is found
        # it means no PUBCHEM record was found, therefore a flag is set.
        for compound in self.compounds_list:
            key: Callable = compound._chemical_ids_url
            try:
                compound._set_chemical_ids_pubresponse(data[key])
                compound.compound_in_PUBCHEM = True
            except KeyError:
                compound.compound_in_PUBCHEM = False

        print("\nFetching safety and vendors data")
        url_fetcher: Callable

        for compound in self.compounds_list:
            for url_fetcher in compound._fetch_pairs.keys():
                if compound.compound_in_PUBCHEM:
                    url_requests.append(url_fetcher)

        data = PUBCHEM_load_balancer(url_requests, attempts)

        # relies on getter/setter dictionary in each compound object to assign
        # the output from PUBCHEM_load_balancer to the correct function.
        for compound in self.compounds_list:
            keys: KeysView = compound._fetch_pairs.keys()
            for key in keys:
                setter = compound._fetch_pairs[key]
                try:
                    response: Any = data[key]  # JSON
                    setter(response)
                except KeyError:
                    pass
        print("Elapsed: ", time.time() - initial_time)

    def add(self, compounds_list: list[Compound]) -> None:
        """Add a list of Compound objects to the library.

        The function appends a list of Compound objects to the library and
        register them to keep track of the properties.

        Parameters
        ----------
        compounds_list : list[Compound]
            A list of Compound objects.

        Raises
        ------
        TypeError
            Raised if anything else than a list of Compound objects is provided.

        Returns
        -------
        None

        """
        if type(compounds_list) is not list:
            raise TypeError(
                f"add method requires a list as input, while {type(compounds_list)} was provided."
            )

        self.compounds_list.extend(compounds_list)

        compound: Compound
        user_property: str

        for compound in compounds_list:
            compound._linked_libraries.append(self)

            for user_property in compound._user_properties.__dict__:
                self._register_property(user_property)

    def read(self, sdf_filepath: str) -> None:
        """Load an sdf file into a library.

        Molecule files are loaded into a Library object. Properties are
        discarded.

        Parameters
        ----------
        sdf_filepath : str
            Path of the sdf file.

        Returns
        -------
        None

        """
        suppl: Chem.SDMolSupplier = Chem.SDMolSupplier(sdf_filepath)

        mol: rdkit.Chem.rdchem.Mol
        compound: Compound

        for mol in suppl:
            try:
                self.compounds_list.append(Compound(rdkitmol=mol, autofetch=False))
            except ValueError:
                print("ERROR: rdkit could not parse the current molecule. Skipping.")
                continue

        for compound in self.compounds_list:
            compound._linked_libraries.append(self)

    def save(self, filename: str, file_format: str = "sdf") -> None:
        """Export the library to an SDF file.

        Compound objects and their properties are exported to an SDF file.

        Parameters
        ----------
        filename : str
            Name of the output file.
        file_format : str, optional
            File format for the output file. The default is "sdf".

        Raises
        ------
        NotImplementedError
            Raised in case a different file format than sdf is requested.

        Returns
        -------
        None
        """
        if file_format != "sdf":
            raise NotImplementedError(
                "No other file format other than SDF is supported at" + "the current time"
            )

        sdf_out: Chem.SDWriter = Chem.SDWriter(filename)

        # output_molecules = []

        compound: Compound

        for compound in self.compounds_list:
            # avoids having replicated data in memory...
            if compound.molecule is not None:
                molecule: rdkit.Chem.rdchem.Mol = copy.copy(compound.molecule)

            else:
                molecule = Chem.rdchem.Mol()

            molecule.SetProp("InChI", str(compound.chemical_ids.InChI))
            molecule.SetProp("InChIKey", str(compound.chemical_ids.InChIKey))
            molecule.SetProp("SMILES", str(compound.chemical_ids.SMILES))
            molecule.SetProp("IUPAC", str(compound.chemical_ids.IUPAC))

            property_name: str

            for property_name in self._registered_properties:
                try:
                    property_value: Any = compound._user_properties.__getattribute__(
                        property_name
                    )
                except AttributeError:
                    property_value = None

                molecule.SetProp(property_name, str(property_value))

            sdf_out.write(molecule)

        sdf_out.close()

    def _register_property(self, user_property: str) -> None:
        """Add a new user property to the registered properties of the library.

        Registered properties will be output to file once exported to SDF.

        Parameters
        ----------
        user_property : str
            Name of the user property to be registered.

        Returns
        -------
        None
        """
        self._registered_properties.add(user_property)

    @property
    def in_pubchem(self) -> None:
        """Yield only compounds that were found on PUBCHEM.

        Yields
        ------
        compound: Compound
            Compound object contained in the library.

        """
        compound: Compound

        for compound in self.compounds_list:
            if compound.compound_in_PUBCHEM:
                yield compound

    def __iter__(self) -> None:
        """Yield compounds of the library.

        Yields
        ------
        compound: Compound
            Compound object contained in the library.

        """
        for compound in self.compounds_list:
            yield compound

    def __getitem__(self, key):
        return self.compounds_list[key]
